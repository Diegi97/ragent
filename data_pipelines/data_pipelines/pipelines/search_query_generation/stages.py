from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import anyio
from prefect import task
from prefect.concurrency.asyncio import concurrency

from data_pipelines.artifacts.ranking import normalize_id, target_rank
from data_pipelines.artifacts.retrieval_queries import RetrievalChunk
from data_pipelines.pipelines.search_query_generation.config import (
    RetrievalQueriesConfig,
)
from data_pipelines.pipelines.search_query_generation.models import (
    FilterReason,
    OutputPaths,
    QueryStatus,
    RetrievalQuery,
)
from data_pipelines.pipelines.search_query_generation.output import (
    append_query_record,
    initialize_output,
)
from data_pipelines.pipelines.search_query_generation.prompts import (
    build_contrastive_narrow_messages,
    build_generate_query_messages,
    parse_contrastive_narrow_response,
    parse_generate_query_response,
)
from data_pipelines.pipelines.search_query_generation.retrieval import (
    chunk_by_source_index,
    count_chunks,
    load_retriever,
)
from data_pipelines.providers.openai import chat_completion
from data_pipelines.tracing import (
    SpanKind,
    set_span_attributes,
    set_span_error,
    set_span_output,
    stage_span,
)
from ragent_core.retrievers.document import Document, RetrievalResult

LLM_CONCURRENCY_LIMIT = "openai-llm"


RETRIEVER_CONCURRENCY_LIMIT = "local-retriever"


def _load_row(config: RetrievalQueriesConfig, row_index: int) -> Document:
    row = chunk_by_source_index(config, row_index)
    if row is None:
        raise RuntimeError(
            f"Turbopuffer source_index {row_index} did not resolve to one chunk."
        )
    return row


def _retrieve(config: RetrievalQueriesConfig, query: str) -> Sequence[RetrievalResult]:
    top_k = max(config.round_trip_top_k, config.candidate_mining_top_k)
    return load_retriever(config).retrieve(
        query,
        table_name=config.table_name,
        top_k=top_k,
    )


@task(name="read-chunk-row-count", retries=0, persist_result=False)
async def read_row_count(config: RetrievalQueriesConfig) -> int:
    return await anyio.to_thread.run_sync(count_chunks, config)


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
        SpanKind.CHAIN,
        {"sample_index": sample_index, "row_index": row_index},
        attributes={"sample.index": sample_index, "row.index": row_index},
    ) as span:
        row = await anyio.to_thread.run_sync(_load_row, config, row_index)
        positive = RetrievalChunk.from_document(row)
        retrieval_query = RetrievalQuery(
            query="",
            doc_id=positive.id,
            positive=positive,
            status=QueryStatus.SAMPLED,
            metadata={
                **metadata,
                "table_name": config.table_name,
                "logical_namespace": config.logical_namespace,
            },
        )
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
        SpanKind.LLM,
        retrieval_query.to_trace_dict(),
        attributes={
            "sample.index": sample_index,
            "row.index": row_index,
            "llm.model_name": config.generator_model,
        },
    ) as span:
        if retrieval_query.positive is None:
            updated_query = retrieval_query.failed(
                QueryStatus.FAILED, "Missing positive chunk."
            )
            set_span_error(
                span, updated_query.failure_reason or "Missing positive chunk."
            )
            set_span_output(span, updated_query.to_trace_dict())
            return updated_query

        async with concurrency(LLM_CONCURRENCY_LIMIT, strict=True):
            completion = await chat_completion(
                build_generate_query_messages(retrieval_query.positive),
                config.generator_model,
            )
        set_span_attributes(span, completion.to_trace_attributes())
        try:
            query = parse_generate_query_response(completion.content)
        except Exception as exc:
            updated_query = retrieval_query.failed(
                QueryStatus.FAILED, f"generate_query: {exc}"
            )
            set_span_error(span, str(exc))
        else:
            updated_query = replace(
                retrieval_query,
                query=query,
                status=QueryStatus.GENERATED,
                failure_reason=None,
                reason_code=None,
                metadata={
                    **retrieval_query.metadata,
                    "generator_model": completion.model,
                },
            )
        set_span_output(span, updated_query.to_trace_dict())
        return updated_query


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
        SpanKind.RETRIEVER,
        retrieval_query.to_trace_dict(),
        attributes={"sample.index": sample_index, "row.index": row_index},
    ) as span:
        if retrieval_query.status != QueryStatus.GENERATED:
            set_span_output(span, retrieval_query.to_trace_dict())
            return retrieval_query

        async with concurrency(RETRIEVER_CONCURRENCY_LIMIT, strict=True):
            results = await anyio.to_thread.run_sync(
                _retrieve,
                config,
                retrieval_query.query,
            )
        round_trip_rank = target_rank(
            [retrieval_result.id for retrieval_result in results],
            retrieval_query.doc_id,
        )
        retrieved_ids = [retrieval_result.id for retrieval_result in results]
        if round_trip_rank is None or round_trip_rank > config.round_trip_top_k:
            reason = (
                "Positive chunk was not retrieved in hybrid top "
                f"{config.round_trip_top_k}."
            )
            updated_query = retrieval_query.failed(
                QueryStatus.FILTERED,
                reason,
                reason_code=FilterReason.ROUND_TRIP_MISS,
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
                RetrievalChunk.from_result(candidate, rank)
                for rank, candidate in enumerate(candidate_results, start=1)
                if normalize_id(candidate.id) != normalize_id(retrieval_query.doc_id)
            )
            updated_query = replace(
                retrieval_query,
                candidates=candidates,
                status=QueryStatus.MINED,
                failure_reason=None,
                reason_code=None,
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
        set_span_output(span, updated_query.to_trace_dict())
        return updated_query


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
        SpanKind.LLM,
        retrieval_query.to_trace_dict(),
        attributes={
            "sample.index": sample_index,
            "row.index": row_index,
            "llm.model_name": config.generator_model,
        },
    ) as span:
        if retrieval_query.status != QueryStatus.MINED:
            set_span_output(span, retrieval_query.to_trace_dict())
            return retrieval_query
        if retrieval_query.positive is None:
            updated_query = retrieval_query.failed(
                QueryStatus.FAILED, "Missing positive chunk."
            )
            set_span_error(
                span, updated_query.failure_reason or "Missing positive chunk."
            )
            set_span_output(span, updated_query.to_trace_dict())
            return updated_query

        candidates = retrieval_query.candidates[: config.contrastive_candidate_count]
        async with concurrency(LLM_CONCURRENCY_LIMIT, strict=True):
            completion = await chat_completion(
                build_contrastive_narrow_messages(
                    retrieval_query.query,
                    retrieval_query.positive,
                    candidates,
                ),
                config.generator_model,
            )
        set_span_attributes(span, completion.to_trace_attributes())
        try:
            keep, query, hard_negative_ids = parse_contrastive_narrow_response(
                completion.content
            )
        except Exception as exc:
            updated_query = retrieval_query.failed(
                QueryStatus.FAILED,
                f"contrastive_narrow: {exc}",
            )
            set_span_error(span, str(exc))
        else:
            if not keep:
                updated_query = retrieval_query.failed(
                    QueryStatus.FILTERED,
                    "LLM marked contrastive sample as not keepable.",
                    reason_code=FilterReason.CONTRASTIVE_REJECTION,
                )
            else:
                hard_negatives = retrieval_query.resolve_hard_negatives(
                    hard_negative_ids, candidates
                )
                updated_query = replace(
                    retrieval_query,
                    query=query,
                    hard_negatives=hard_negatives,
                    status=QueryStatus.READY,
                    failure_reason=None,
                    reason_code=None,
                    metadata={
                        **retrieval_query.metadata,
                        "contrastive_candidate_count": len(candidates),
                        "hard_negative_count": len(hard_negatives),
                    },
                )
        set_span_attributes(
            span,
            {
                "query.status": updated_query.status,
                "failure.reason": updated_query.failure_reason,
            },
        )
        set_span_output(span, updated_query.to_trace_dict())
        return updated_query


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
        SpanKind.CHAIN,
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
