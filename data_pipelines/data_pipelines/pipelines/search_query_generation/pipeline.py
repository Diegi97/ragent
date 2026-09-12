import asyncio
import random
from dataclasses import replace
from typing import Any

import anyio
from prefect import flow, get_run_logger
from prefect.client.orchestration import get_client
from prefect.runtime import flow_run

from data_pipelines.pipelines.search_query_generation.config import (
    RetrievalQueriesConfig,
)
from data_pipelines.pipelines.search_query_generation.models import (
    ObjectPipelineError,
    ObjectRunSummary,
    OutputPaths,
    QueryStatus,
    RetrievalQuery,
)
from data_pipelines.pipelines.search_query_generation.output import (
    write_metadata,
)
from data_pipelines.pipelines.search_query_generation.stages import (
    LLM_CONCURRENCY_LIMIT,
    RETRIEVER_CONCURRENCY_LIMIT,
    append_record,
    contrastive_narrow,
    generate_query,
    initialize_jsonl_output,
    read_row_count,
    retrieve_candidates,
    sample_chunk,
)
from data_pipelines.timestamps import TimestampPrecision, utc_timestamp
from data_pipelines.tracing import (
    configure_tracing,
    object_trace,
    set_span_attributes,
)


def search_query_object_name(
    batch_timestamp: str,
    sample_index: int,
    row_index: int,
) -> str:
    return f"search-query-generation-{batch_timestamp}-{sample_index}-{row_index}"


def reserve_indices(row_count: int, sample_count: int, seed: int) -> list[int]:
    """Return a deterministic, unique set of valid remote source indices."""
    if row_count < 0:
        raise ValueError("row_count cannot be negative")
    bounded_count = min(max(sample_count, 0), row_count)
    return random.Random(seed).sample(range(row_count), bounded_count)


async def _upsert_concurrency_limits(config: RetrievalQueriesConfig) -> None:
    async with get_client() as client:
        await client.upsert_global_concurrency_limit_by_name(
            LLM_CONCURRENCY_LIMIT,
            config.llm_concurrency,
        )
        await client.upsert_global_concurrency_limit_by_name(
            RETRIEVER_CONCURRENCY_LIMIT,
            config.retriever_concurrency,
        )


def _initial_query(metadata: dict[str, Any]) -> RetrievalQuery:
    return RetrievalQuery(
        query="",
        doc_id=None,
        status=QueryStatus.FAILED,
        failure_reason="Object did not reach sampling.",
        metadata=metadata,
    )


@flow(
    name="search-query-generation-object",
    flow_run_name=(
        "search-query-generation-{batch_timestamp}-{sample_index}-{row_index}"
    ),
    retries=0,
    persist_result=False,
)
async def search_query_generation_flow(
    config: RetrievalQueriesConfig,
    sample_index: int,
    row_index: int,
    paths: OutputPaths,
    batch_run_id: str,
    batch_timestamp: str,
) -> ObjectRunSummary:
    child_run_id = str(flow_run.id)
    object_id = search_query_object_name(batch_timestamp, sample_index, row_index)
    base_metadata = {
        "object_id": object_id,
        "batch_timestamp": batch_timestamp,
        "sample_index": sample_index,
        "row_index": row_index,
        "prefect_flow_run_id": child_run_id,
        "prefect_batch_flow_run_id": batch_run_id,
    }
    current_query = _initial_query(base_metadata)
    current_stage = "sample_chunk"
    pending_error: ObjectPipelineError | None = None

    with object_trace(
        object_id,
        {"sample_index": sample_index, "row_index": row_index},
        attributes={
            "object.id": object_id,
            "batch.timestamp": batch_timestamp,
            "sample.index": sample_index,
            "row.index": row_index,
            "prefect.batch_flow_run_id": batch_run_id,
            "prefect.child_flow_run_id": child_run_id,
            "llm.model_name": config.generator_model,
        },
    ) as root:
        current_query = replace(
            current_query,
            metadata={**current_query.metadata, "phoenix_trace_id": root.trace_id},
        )
        try:
            current_query = await sample_chunk(
                config,
                sample_index,
                row_index,
                current_query.metadata,
                root.carrier,
            )
            current_stage = "generate_query"
            current_query = await generate_query(
                current_query,
                config,
                sample_index,
                row_index,
                root.carrier,
            )
            current_stage = "retrieve_candidates"
            current_query = await retrieve_candidates(
                current_query,
                config,
                sample_index,
                row_index,
                root.carrier,
            )
            current_stage = "contrastive_narrow"
            current_query = await contrastive_narrow(
                current_query,
                config,
                sample_index,
                row_index,
                root.carrier,
            )
        except Exception as exc:
            logger = get_run_logger()
            logger.exception(
                "Unexpected failure for object %s in %s",
                object_id,
                current_stage,
            )
            current_query = current_query.failed(
                QueryStatus.FAILED,
                f"{current_stage}: {type(exc).__name__}: {exc}",
                crashed=True,
                crashed_stage=current_stage,
                exception_type=type(exc).__name__,
            )
            root.mark_error(exc)
            pending_error = ObjectPipelineError(
                ObjectRunSummary(
                    object_id=object_id,
                    sample_index=sample_index,
                    row_index=row_index,
                    status=QueryStatus.FAILED,
                    phoenix_trace_id=root.trace_id,
                    crashed=True,
                    error=current_query.failure_reason,
                )
            )

        record_path: str | None = None
        try:
            current_stage = "append_record"
            record_path = await append_record(
                current_query,
                paths,
                config.hard_negatives_per_query,
                sample_index,
                row_index,
                root.carrier,
            )
        except Exception as exc:
            root.mark_error(exc)
            if pending_error is None:
                current_query = current_query.failed(
                    QueryStatus.FAILED,
                    f"append_record: {type(exc).__name__}: {exc}",
                    crashed=True,
                    crashed_stage="append_record",
                    exception_type=type(exc).__name__,
                )
                pending_error = ObjectPipelineError(
                    ObjectRunSummary(
                        object_id=object_id,
                        sample_index=sample_index,
                        row_index=row_index,
                        status=QueryStatus.FAILED,
                        phoenix_trace_id=root.trace_id,
                        crashed=True,
                        error=current_query.failure_reason,
                    )
                )

        summary = ObjectRunSummary(
            object_id=object_id,
            sample_index=sample_index,
            row_index=row_index,
            status=current_query.status,
            phoenix_trace_id=root.trace_id,
            crashed=pending_error is not None,
            error=pending_error.summary.error if pending_error else None,
            record_path=record_path,
        )
        set_span_attributes(
            root.span,
            {
                "document.id": (
                    str(current_query.doc_id)
                    if current_query.doc_id is not None
                    else None
                ),
                "query.status": current_query.status,
                "failure.reason": current_query.failure_reason,
            },
        )
        if pending_error is not None:
            pending_error.summary = summary
            root.set_output(
                {
                    "status": summary.status,
                    "crashed": True,
                    "record_path": record_path,
                    "error": summary.error,
                }
            )
        else:
            root.set_output(
                {
                    "status": summary.status,
                    "crashed": False,
                    "record_path": record_path,
                }
            )

    await anyio.to_thread.run_sync(configure_tracing().force_flush)
    if pending_error is not None:
        raise pending_error
    return summary


def _summary_for_outcome(
    outcome: ObjectRunSummary | BaseException,
    batch_timestamp: str,
    sample_index: int,
    row_index: int,
) -> ObjectRunSummary:
    if isinstance(outcome, ObjectRunSummary):
        return outcome
    candidate: BaseException | None = outcome
    seen: set[int] = set()
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        if isinstance(candidate, ObjectPipelineError):
            return candidate.summary
        candidate = candidate.__cause__ or candidate.__context__
    return ObjectRunSummary(
        object_id=search_query_object_name(
            batch_timestamp,
            sample_index,
            row_index,
        ),
        sample_index=sample_index,
        row_index=row_index,
        status=QueryStatus.FAILED,
        phoenix_trace_id="",
        crashed=True,
        error=f"{type(outcome).__name__}: {outcome}",
    )


@flow(
    name="search-query-generation-batch",
    flow_run_name="search-query-generation-batch-{config.table_name}",
    retries=0,
    persist_result=False,
)
async def search_query_generation_batch_flow(
    config: RetrievalQueriesConfig,
) -> dict[str, Any]:
    logger = get_run_logger()
    batch_run_id = str(flow_run.id)
    batch_timestamp = utc_timestamp(TimestampPrecision.MICROSECONDS)
    tracing = configure_tracing()
    await _upsert_concurrency_limits(config)

    row_count = await read_row_count(config)
    if row_count == 0 and config.num_queries > 0:
        raise ValueError("Cannot generate queries from an empty source corpus.")
    row_indices = reserve_indices(row_count, config.num_queries, config.seed)
    if len(row_indices) < config.num_queries:
        logger.warning(
            "Requested %d objects but table contains only %d rows; reserving %d.",
            config.num_queries,
            row_count,
            len(row_indices),
        )
    paths = await initialize_jsonl_output(config, batch_run_id)

    outcomes = await asyncio.gather(
        *(
            search_query_generation_flow(
                config,
                sample_index,
                row_index,
                paths,
                batch_run_id,
                batch_timestamp,
            )
            for sample_index, row_index in enumerate(row_indices)
        ),
        return_exceptions=True,
    )
    summaries = [
        _summary_for_outcome(
            outcome, batch_timestamp, sample_index, row_indices[sample_index]
        )
        for sample_index, outcome in enumerate(outcomes)
    ]
    metadata = await anyio.to_thread.run_sync(
        write_metadata,
        config,
        paths,
        batch_run_id,
        batch_timestamp,
        tracing.project_name,
        summaries,
    )
    await anyio.to_thread.run_sync(tracing.force_flush)

    crashed = [summary for summary in summaries if summary.crashed]
    if crashed:
        raise RuntimeError(
            f"{len(crashed)} search-query-generation object flow(s) crashed unexpectedly; "
            f"metadata was written to {paths.metadata_path}."
        )
    return metadata
