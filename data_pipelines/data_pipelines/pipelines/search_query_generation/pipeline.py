import asyncio
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import anyio
from prefect import flow, get_run_logger, task
from prefect.client.orchestration import get_client
from prefect.concurrency.asyncio import concurrency
from prefect.runtime import flow_run

from data_pipelines.pipelines.search_query_generation.config import (
    RetrievalQueriesConfig,
)
from data_pipelines.pipelines.search_query_generation.models import (
    ObjectPipelineError,
    ObjectRunSummary,
    OutputPaths,
    RetrievalChunk,
    RetrievalQuery,
)
from data_pipelines.pipelines.search_query_generation.output import (
    append_query_record,
    initialize_output,
    write_metadata,
)
from data_pipelines.pipelines.search_query_generation.prompts import (
    build_contrastive_narrow_messages,
    build_generate_query_messages,
    parse_contrastive_narrow_response,
    parse_generate_query_response,
)
from data_pipelines.pipelines.search_query_generation.services import (
    LLMCompletion,
    chat_completion,
    chunks_table,
    load_retriever,
)
from data_pipelines.tracing import (
    configure_tracing,
    get_tracing,
    object_trace,
    set_span_attributes,
    set_span_error,
    set_span_output,
    stage_span,
)

LLM_CONCURRENCY_LIMIT = "openai-llm"
RETRIEVER_CONCURRENCY_LIMIT = "local-retriever"
DOCUMENT_ID_KEY = "document_id"


def utc_run_timestamp() -> str:
    """Return a sortable UTC timestamp safe for Prefect and Phoenix names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def search_query_object_name(
    batch_timestamp: str,
    sample_index: int,
    row_index: int,
) -> str:
    return f"search-query-generation-{batch_timestamp}-{sample_index}-{row_index}"


def reserve_indices(row_count: int, sample_count: int, seed: int) -> list[int]:
    """Return a deterministic, unique set of valid LanceDB row indices."""
    if row_count < 0:
        raise ValueError("row_count cannot be negative")
    bounded_count = min(max(sample_count, 0), row_count)
    return random.Random(seed).sample(range(row_count), bounded_count)


def _same_id(left: Any, right: Any) -> bool:
    return str(left) == str(right)


def _chunk_from_row(row: Mapping[str, Any]) -> RetrievalChunk | None:
    chunk_id = row.get("id")
    if chunk_id is None or (isinstance(chunk_id, str) and not chunk_id.strip()):
        return None
    return RetrievalChunk(
        id=chunk_id,
        title=str(row.get("title") or ""),
        text=str(row.get("content") or ""),
        document_id=row.get("document_id"),
        metadata=dict(row.get("metadata") or {}),
    )


def _candidate_from_result(
    result: Any,
    rank: int,
) -> RetrievalChunk:
    metadata = dict(result.metadata or {})
    return RetrievalChunk(
        id=result.id,
        title=result.title,
        text=result.content,
        document_id=metadata.get(DOCUMENT_ID_KEY),
        metadata=metadata,
        score=float(result.score),
        sources=("hybrid",),
        source_ranks={"hybrid": rank},
    )


def _completion_attributes(completion: LLMCompletion) -> dict[str, Any]:
    return {
        "llm.model_name": completion.model,
        "llm.token_count.prompt": completion.prompt_tokens,
        "llm.token_count.completion": completion.completion_tokens,
        "llm.token_count.total": completion.total_tokens,
    }


def _load_row(config: RetrievalQueriesConfig, row_index: int) -> Mapping[str, Any]:
    rows = (
        chunks_table(config)
        .to_lance()
        .take(
            [row_index],
            columns=["id", "title", "content", "metadata", "document_id"],
        )
        .to_pylist()
    )
    if len(rows) != 1:
        raise RuntimeError(f"LanceDB row {row_index} did not resolve to one chunk.")
    return rows[0]


def _retrieve(config: RetrievalQueriesConfig, query: str) -> Sequence[Any]:
    top_k = max(config.round_trip_top_k, config.candidate_mining_top_k)
    return load_retriever(config).retrieve(
        query,
        table_name=config.table_name,
        top_k=top_k,
    )


@task(name="read-chunk-row-count", retries=0, persist_result=False)
async def read_row_count(config: RetrievalQueriesConfig) -> int:
    return await anyio.to_thread.run_sync(lambda: chunks_table(config).count_rows())


@task(name="initialize-jsonl-output", retries=0, persist_result=False)
async def initialize_jsonl_output(
    config: RetrievalQueriesConfig,
    batch_run_id: str,
) -> OutputPaths:
    return await anyio.to_thread.run_sync(initialize_output, config, batch_run_id)


@task(
    name="sample-chunk",
    task_run_name="sample-{sample_index}-{row_index}",
    retries=0,
    persist_result=False,
)
async def sample_chunk(
    config: RetrievalQueriesConfig,
    sample_index: int,
    row_index: int,
    metadata: dict[str, Any],
    trace_carrier: dict[str, str],
) -> RetrievalQuery:
    with stage_span(
        trace_carrier,
        "sample_chunk",
        "CHAIN",
        {"sample_index": sample_index, "row_index": row_index},
        attributes={"sample.index": sample_index, "row.index": row_index},
    ) as span:
        row = await anyio.to_thread.run_sync(_load_row, config, row_index)
        positive = _chunk_from_row(row)
        retrieval_query = RetrievalQuery(
            query="",
            doc_id=positive.id if positive is not None else None,
            positive=positive,
            status="sampled" if positive is not None else "failed",
            failure_reason=None if positive is not None else "Missing positive chunk.",
            metadata={
                **metadata,
                "table_name": config.table_name,
                "retriever_namespace": config.retriever_namespace,
            },
        )
        if positive is None:
            set_span_error(span, "Missing positive chunk.")
        else:
            set_span_attributes(span, {"document.id": str(positive.id)})
        set_span_output(span, retrieval_query.to_trace_dict())
        return retrieval_query


@task(
    name="generate-query",
    task_run_name="generate-{sample_index}-{row_index}",
    retries=0,
    persist_result=False,
)
async def generate_query(
    retrieval_query: RetrievalQuery,
    config: RetrievalQueriesConfig,
    sample_index: int,
    row_index: int,
    trace_carrier: dict[str, str],
) -> RetrievalQuery:
    with stage_span(
        trace_carrier,
        "generate_query",
        "LLM",
        retrieval_query.to_trace_dict(),
        attributes={
            "sample.index": sample_index,
            "row.index": row_index,
            "llm.model_name": config.generator_model,
        },
    ) as span:
        if retrieval_query.positive is None:
            result = retrieval_query.failed("failed", "Missing positive chunk.")
            set_span_error(span, result.failure_reason or "Missing positive chunk.")
            set_span_output(span, result.to_trace_dict())
            return result

        async with concurrency(LLM_CONCURRENCY_LIMIT, strict=True):
            completion = await chat_completion(
                build_generate_query_messages(retrieval_query.positive),
                config,
            )
        set_span_attributes(span, _completion_attributes(completion))
        try:
            query = parse_generate_query_response(completion.content)
        except Exception as exc:
            result = retrieval_query.failed("failed", f"generate_query: {exc}")
            set_span_error(span, str(exc))
        else:
            result = replace(
                retrieval_query,
                query=query,
                status="generated",
                failure_reason=None,
                metadata={
                    **retrieval_query.metadata,
                    "generator_model": completion.model,
                },
            )
        set_span_output(span, result.to_trace_dict())
        return result


@task(
    name="retrieve-candidates",
    task_run_name="retrieve-{sample_index}-{row_index}",
    retries=0,
    persist_result=False,
)
async def retrieve_candidates(
    retrieval_query: RetrievalQuery,
    config: RetrievalQueriesConfig,
    sample_index: int,
    row_index: int,
    trace_carrier: dict[str, str],
) -> RetrievalQuery:
    with stage_span(
        trace_carrier,
        "retrieve_candidates",
        "RETRIEVER",
        retrieval_query.to_trace_dict(),
        attributes={"sample.index": sample_index, "row.index": row_index},
    ) as span:
        if retrieval_query.status != "generated":
            set_span_output(span, retrieval_query.to_trace_dict())
            return retrieval_query

        async with concurrency(RETRIEVER_CONCURRENCY_LIMIT, strict=True):
            results = await anyio.to_thread.run_sync(
                _retrieve,
                config,
                retrieval_query.query,
            )
        round_trip_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if _same_id(result.id, retrieval_query.doc_id)
            ),
            None,
        )
        retrieved_ids = [result.id for result in results]
        if round_trip_rank is None or round_trip_rank > config.round_trip_top_k:
            reason = (
                "Positive chunk was not retrieved in hybrid top "
                f"{config.round_trip_top_k}."
            )
            result = retrieval_query.failed(
                "filtered",
                reason,
                retrieval_top_k=max(
                    config.round_trip_top_k,
                    config.candidate_mining_top_k,
                ),
                round_trip_rank=round_trip_rank,
                retrieved_ids=retrieved_ids,
            )
            set_span_attributes(
                span,
                {
                    "retrieval.documents": len(results),
                    "retrieval.round_trip_rank": round_trip_rank,
                    "failure.reason": reason,
                },
            )
        else:
            candidate_results = results[: config.candidate_mining_top_k]
            candidates = tuple(
                _candidate_from_result(candidate, rank)
                for rank, candidate in enumerate(candidate_results, start=1)
                if not _same_id(candidate.id, retrieval_query.doc_id)
            )
            result = replace(
                retrieval_query,
                candidates=candidates,
                status="mined",
                failure_reason=None,
                metadata={
                    **retrieval_query.metadata,
                    "retrieval_top_k": max(
                        config.round_trip_top_k,
                        config.candidate_mining_top_k,
                    ),
                    "round_trip_rank": round_trip_rank,
                    "mined_candidate_count": len(candidates),
                },
            )
            set_span_attributes(
                span,
                {
                    "retrieval.documents": len(results),
                    "retrieval.round_trip_rank": round_trip_rank,
                },
            )
        set_span_output(span, result.to_trace_dict())
        return result


@task(
    name="contrastive-narrow",
    task_run_name="narrow-{sample_index}-{row_index}",
    retries=0,
    persist_result=False,
)
async def contrastive_narrow(
    retrieval_query: RetrievalQuery,
    config: RetrievalQueriesConfig,
    sample_index: int,
    row_index: int,
    trace_carrier: dict[str, str],
) -> RetrievalQuery:
    with stage_span(
        trace_carrier,
        "contrastive_narrow",
        "LLM",
        retrieval_query.to_trace_dict(),
        attributes={
            "sample.index": sample_index,
            "row.index": row_index,
            "llm.model_name": config.generator_model,
        },
    ) as span:
        if retrieval_query.status != "mined":
            set_span_output(span, retrieval_query.to_trace_dict())
            return retrieval_query
        if retrieval_query.positive is None:
            result = retrieval_query.failed("failed", "Missing positive chunk.")
            set_span_error(span, result.failure_reason or "Missing positive chunk.")
            set_span_output(span, result.to_trace_dict())
            return result

        candidates = retrieval_query.candidates[: config.contrastive_candidate_count]
        async with concurrency(LLM_CONCURRENCY_LIMIT, strict=True):
            completion = await chat_completion(
                build_contrastive_narrow_messages(
                    retrieval_query.query,
                    retrieval_query.positive,
                    candidates,
                ),
                config,
            )
        set_span_attributes(span, _completion_attributes(completion))
        try:
            keep, query, hard_negative_ids = parse_contrastive_narrow_response(
                completion.content
            )
        except Exception as exc:
            result = retrieval_query.failed(
                "failed",
                f"contrastive_narrow: {exc}",
            )
            set_span_error(span, str(exc))
        else:
            if not keep:
                result = retrieval_query.failed(
                    "filtered",
                    "LLM marked contrastive sample as not keepable.",
                )
            else:
                candidate_by_id = {
                    str(candidate.id): candidate for candidate in candidates
                }
                hard_negatives = tuple(
                    candidate_by_id[chunk_id]
                    for chunk_id in hard_negative_ids
                    if chunk_id in candidate_by_id
                )
                result = replace(
                    retrieval_query,
                    query=query,
                    hard_negatives=hard_negatives,
                    status="ready",
                    failure_reason=None,
                    metadata={
                        **retrieval_query.metadata,
                        "contrastive_candidate_count": len(candidates),
                        "hard_negative_count": len(hard_negatives),
                    },
                )
        set_span_attributes(
            span,
            {
                "query.status": result.status,
                "failure.reason": result.failure_reason,
            },
        )
        set_span_output(span, result.to_trace_dict())
        return result


@task(
    name="append-record",
    task_run_name="append-{sample_index}-{row_index}",
    retries=0,
    persist_result=False,
)
async def append_record(
    retrieval_query: RetrievalQuery,
    paths: OutputPaths,
    hard_negatives_per_query: int,
    sample_index: int,
    row_index: int,
    trace_carrier: dict[str, str],
) -> str:
    with stage_span(
        trace_carrier,
        "append_record",
        "CHAIN",
        retrieval_query.to_trace_dict(),
        attributes={"sample.index": sample_index, "row.index": row_index},
    ) as span:
        destination = await anyio.to_thread.run_sync(
            append_query_record,
            retrieval_query,
            paths,
            hard_negatives_per_query,
        )
        set_span_output(
            span,
            {"path": str(destination), "status": retrieval_query.status},
        )
        return str(destination)


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
        status="failed",
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
    current = _initial_query(base_metadata)
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
        current = replace(
            current,
            metadata={**current.metadata, "phoenix_trace_id": root.trace_id},
        )
        try:
            current = await sample_chunk(
                config,
                sample_index,
                row_index,
                current.metadata,
                root.carrier,
            )
            current_stage = "generate_query"
            current = await generate_query(
                current,
                config,
                sample_index,
                row_index,
                root.carrier,
            )
            current_stage = "retrieve_candidates"
            current = await retrieve_candidates(
                current,
                config,
                sample_index,
                row_index,
                root.carrier,
            )
            current_stage = "contrastive_narrow"
            current = await contrastive_narrow(
                current,
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
            current = current.failed(
                "failed",
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
                    status="failed",
                    phoenix_trace_id=root.trace_id,
                    crashed=True,
                    error=current.failure_reason,
                )
            )

        record_path: str | None = None
        try:
            current_stage = "append_record"
            record_path = await append_record(
                current,
                paths,
                config.hard_negatives_per_query,
                sample_index,
                row_index,
                root.carrier,
            )
        except Exception as exc:
            root.mark_error(exc)
            if pending_error is None:
                current = current.failed(
                    "failed",
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
                        status="failed",
                        phoenix_trace_id=root.trace_id,
                        crashed=True,
                        error=current.failure_reason,
                    )
                )

        summary = ObjectRunSummary(
            object_id=object_id,
            sample_index=sample_index,
            row_index=row_index,
            status=current.status,
            phoenix_trace_id=root.trace_id,
            crashed=pending_error is not None,
            error=pending_error.summary.error if pending_error else None,
            record_path=record_path,
        )
        set_span_attributes(
            root.span,
            {
                "document.id": (
                    str(current.doc_id) if current.doc_id is not None else None
                ),
                "query.status": current.status,
                "failure.reason": current.failure_reason,
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

    get_tracing().force_flush()
    if pending_error is not None:
        raise pending_error
    return summary


def _summary_from_exception(
    error: BaseException,
    batch_run_id: str,
    batch_timestamp: str,
    sample_index: int,
    row_index: int,
) -> ObjectRunSummary:
    candidate: BaseException | None = error
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
        status="failed",
        phoenix_trace_id="",
        crashed=True,
        error=f"{type(error).__name__}: {error}",
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
    batch_timestamp = utc_run_timestamp()
    tracing = configure_tracing()
    await _upsert_concurrency_limits(config)

    row_count = await read_row_count(config)
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
        (
            outcome
            if isinstance(outcome, ObjectRunSummary)
            else _summary_from_exception(
                outcome,
                batch_run_id,
                batch_timestamp,
                sample_index,
                row_indices[sample_index],
            )
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
    tracing.force_flush()

    crashed = [summary for summary in summaries if summary.crashed]
    if crashed:
        raise RuntimeError(
            f"{len(crashed)} search-query-generation object flow(s) crashed unexpectedly; "
            f"metadata was written to {paths.metadata_path}."
        )
    return metadata
