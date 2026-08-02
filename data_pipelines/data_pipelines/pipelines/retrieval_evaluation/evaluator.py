from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_pipelines.pipelines.retrieval_evaluation.config import (
    RetrievalEvaluationConfig,
    SearchType,
)
from data_pipelines.pipelines.retrieval_evaluation.metrics import (
    aggregate_metrics,
    deduplicate_ids,
    metrics_for_rank,
    normalize_id,
    target_rank,
)
from data_pipelines.pipelines.retrieval_evaluation.models import (
    DatasetContext,
    EvaluationSummary,
    QueryRecord,
)
from ragent_core.retrievers.retriever import LanceDBRetriever

logger = logging.getLogger(__name__)


def _valid_id(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def load_dataset_context(input_directory: Path) -> DatasetContext:
    queries_path = input_directory / "queries.jsonl"
    metadata_path = input_directory / "metadata.json"
    if not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {input_directory}")
    if not queries_path.is_file():
        raise ValueError(f"Missing queries.jsonl in {input_directory}")
    if not metadata_path.is_file():
        raise ValueError(f"Missing metadata.json in {input_directory}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read valid metadata from {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict) or not isinstance(metadata.get("config"), dict):
        raise ValueError("metadata.json must contain a 'config' object.")

    source_config = metadata["config"]
    table_name = source_config.get("table_name")
    namespace = source_config.get("retriever_namespace")
    db_uri = source_config.get("lancedb_db_uri")
    if not isinstance(table_name, str) or not table_name.strip():
        raise ValueError("metadata.json config.table_name must be a non-empty string.")
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError(
            "metadata.json config.retriever_namespace must be a non-empty string."
        )
    if not isinstance(db_uri, str) or not db_uri.strip():
        raise ValueError("metadata.json config.lancedb_db_uri must be a path string.")

    return DatasetContext(
        input_directory=input_directory,
        queries_path=queries_path,
        metadata_path=metadata_path,
        table_name=table_name.strip(),
        retriever_namespace=namespace.strip(),
        lancedb_db_uri=Path(db_uri).expanduser().resolve(),
        source_metadata=metadata,
    )


def load_query_records(path: Path) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Query record on {path}:{line_number} must be an object.")
        query = raw.get("query")
        positive = raw.get("positive")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Query record on {path}:{line_number} requires a non-empty query."
            )
        if not isinstance(positive, dict):
            raise ValueError(
                f"Query record on {path}:{line_number} requires a positive object."
            )
        chunk_id = positive.get("id")
        document_id = positive.get("document_id")
        if not _valid_id(chunk_id):
            raise ValueError(
                f"Query record on {path}:{line_number} requires positive.id."
            )
        if not _valid_id(document_id):
            raise ValueError(
                f"Query record on {path}:{line_number} requires positive.document_id."
            )
        metadata = raw.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(
                f"Query record on {path}:{line_number} metadata must be an object."
            )
        records.append(
            QueryRecord(
                index=len(records),
                line_number=line_number,
                query=query.strip(),
                positive_chunk_id=chunk_id,
                positive_document_id=document_id,
                metadata=dict(metadata or {}),
            )
        )
    if not records:
        raise ValueError(f"No query records were found in {path}.")
    return records


@contextmanager
def _lancedb_uri(uri: Path) -> Iterator[None]:
    previous = os.environ.get("LANCEDB_DB_URI")
    os.environ["LANCEDB_DB_URI"] = str(uri)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LANCEDB_DB_URI", None)
        else:
            os.environ["LANCEDB_DB_URI"] = previous


def _load_retriever(
    config: RetrievalEvaluationConfig,
    context: DatasetContext,
) -> LanceDBRetriever:
    use_embeddings = config.search_type is not SearchType.BM25
    with _lancedb_uri(context.lancedb_db_uri):
        return LanceDBRetriever.load_index(
            namespace=context.retriever_namespace,
            model_name=config.embedding_model,
            device=config.device,
            trust_remote_code=config.trust_remote_code,
            reranker_model_name=config.reranker_model if config.reranker else None,
            rerank_threshold=config.reranker_threshold,
            top_rerank=config.reranker_candidate_k,
            rerank_batch_size=config.reranker_batch_size,
            max_seq_length=config.max_seq_length,
            embedding_service_url=(
                config.embedding_service_url if use_embeddings else None
            ),
            reranker_service_url=(
                config.reranker_service_url if config.reranker else None
            ),
            load_embedding_backend=use_embeddings,
        )


def _retrieve(
    retriever: LanceDBRetriever,
    config: RetrievalEvaluationConfig,
    context: DatasetContext,
    query: str,
) -> Sequence[Any]:
    if config.search_type is SearchType.DENSE:
        return retriever.retrieve_dense(
            query,
            table_name=context.table_name,
            top_k=config.top_k,
        )
    if config.search_type is SearchType.BM25:
        return retriever.retrieve_bm25(
            query,
            table_name=context.table_name,
            top_k=config.top_k,
        )
    return retriever.retrieve(
        query,
        table_name=context.table_name,
        top_k=config.top_k,
    )


def _deduplicate_results(results: Sequence[Any]) -> list[Any]:
    seen: set[str] = set()
    deduplicated: list[Any] = []
    for result in results:
        result_id = getattr(result, "id", None)
        if not _valid_id(result_id):
            continue
        normalized = normalize_id(result_id)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(result)
    return deduplicated


def _safe_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _result_document_id(result: Any) -> Any:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    return metadata.get("document_id")


def _result_details(results: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "id": result.id,
            "document_id": _result_document_id(result),
            "score": _safe_score(getattr(result, "score", None)),
            "title": str(getattr(result, "title", "") or ""),
        }
        for rank, result in enumerate(results, start=1)
    ]


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(latencies_ms: Sequence[float], total_ms: float) -> dict[str, Any]:
    return {
        "total_ms": total_ms,
        "mean_query_ms": (
            sum(latencies_ms) / len(latencies_ms) if latencies_ms else None
        ),
        "p50_query_ms": _percentile(latencies_ms, 0.5),
        "p95_query_ms": _percentile(latencies_ms, 0.95),
        "min_query_ms": min(latencies_ms) if latencies_ms else None,
        "max_query_ms": max(latencies_ms) if latencies_ms else None,
    }


def _output_directory(config: RetrievalEvaluationConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reranker_suffix = "_reranked" if config.reranker else ""
    output_directory = config.input_directory / (
        f"{timestamp}_retrieval-evaluation_{config.search_type.value}{reranker_suffix}"
    )
    output_directory.mkdir(exist_ok=False)
    return output_directory


def _write_json(path: Path, value: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as fp:
        json.dump(value, fp, indent=2, ensure_ascii=False, allow_nan=False)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temporary_path, path)


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temporary_path, path)


def _source_provenance(context: DatasetContext) -> dict[str, Any]:
    metadata = context.source_metadata
    source_config = metadata.get("config", {})
    return {
        "input_directory": str(context.input_directory),
        "queries_path": str(context.queries_path),
        "metadata_path": str(context.metadata_path),
        "batch_timestamp": metadata.get("batch_timestamp"),
        "prefect_flow_run_id": metadata.get("prefect_flow_run_id"),
        "generator_model": source_config.get("generator_model"),
        "requested_records": metadata.get("requested_records"),
        "trainable_records": metadata.get("trainable_records"),
    }


def evaluate_retrieval(config: RetrievalEvaluationConfig) -> EvaluationSummary:
    """Evaluate one LanceDB retrieval configuration against generated queries."""
    context = load_dataset_context(config.input_directory)
    queries = load_query_records(context.queries_path)

    load_started = time.perf_counter()
    retriever = _load_retriever(config, context)
    retriever_load_ms = (time.perf_counter() - load_started) * 1000.0

    details: list[dict[str, Any]] = []
    chunk_ranks: list[int | None] = []
    document_ranks: list[int | None] = []
    latencies_ms: list[float] = []
    failed_queries = 0
    evaluation_started = time.perf_counter()

    for record in queries:
        query_started = time.perf_counter()
        try:
            raw_results = _retrieve(retriever, config, context, record.query)
            results = _deduplicate_results(raw_results)
            chunk_ids = [result.id for result in results]
            document_ids = deduplicate_ids(
                _result_document_id(result) for result in results
            )
            chunk_rank = target_rank(chunk_ids, record.positive_chunk_id)
            document_rank = target_rank(document_ids, record.positive_document_id)
            latency_ms = (time.perf_counter() - query_started) * 1000.0
            latencies_ms.append(latency_ms)
            chunk_ranks.append(chunk_rank)
            document_ranks.append(document_rank)
            details.append(
                {
                    "query_index": record.index,
                    "source_line_number": record.line_number,
                    "object_id": record.metadata.get("object_id"),
                    "query": record.query,
                    "positive_chunk_id": record.positive_chunk_id,
                    "positive_document_id": record.positive_document_id,
                    "status": "success",
                    "error": None,
                    "latency_ms": latency_ms,
                    "chunk_rank": chunk_rank,
                    "document_rank": document_rank,
                    "metrics": {
                        "chunk": metrics_for_rank(chunk_rank, config.cutoffs),
                        "document": metrics_for_rank(document_rank, config.cutoffs),
                    },
                    "retrieved_results": _result_details(results),
                }
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - query_started) * 1000.0
            latencies_ms.append(latency_ms)
            failed_queries += 1
            logger.exception("Retrieval failed for query index %d", record.index)
            details.append(
                {
                    "query_index": record.index,
                    "source_line_number": record.line_number,
                    "object_id": record.metadata.get("object_id"),
                    "query": record.query,
                    "positive_chunk_id": record.positive_chunk_id,
                    "positive_document_id": record.positive_document_id,
                    "status": "error",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "latency_ms": latency_ms,
                    "chunk_rank": None,
                    "document_rank": None,
                    "metrics": None,
                    "retrieved_results": [],
                }
            )

    evaluation_ms = (time.perf_counter() - evaluation_started) * 1000.0
    successful_queries = len(queries) - failed_queries
    metrics = {
        "chunk": aggregate_metrics(chunk_ranks, config.cutoffs),
        "document": aggregate_metrics(document_ranks, config.cutoffs),
    }
    output_directory = _output_directory(config)
    summary_path = output_directory / "summary.json"
    details_path = output_directory / "details.jsonl"
    coverage = successful_queries / len(queries)
    summary_record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": _source_provenance(context),
        "resolved_config": {
            **config.model_dump(mode="json"),
            "table_name": context.table_name,
            "retriever_namespace": context.retriever_namespace,
            "lancedb_db_uri": str(context.lancedb_db_uri),
            "embedding_backend": (
                "not_used"
                if config.search_type is SearchType.BM25
                else "service"
                if config.embedding_service_url
                else "local"
            ),
            "reranker_backend": (
                "disabled"
                if not config.reranker
                else "service"
                if config.reranker_service_url
                else "local"
            ),
        },
        "output_directory": str(output_directory),
        "counts": {
            "total_queries": len(queries),
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "coverage": coverage,
        },
        "latency": {
            "retriever_load_ms": retriever_load_ms,
            **_latency_summary(latencies_ms, evaluation_ms),
        },
        "metrics": metrics,
    }
    _write_jsonl(details_path, details)
    _write_json(summary_path, summary_record)
    logger.info("Retrieval evaluation written to %s", output_directory)
    return EvaluationSummary(
        output_directory=output_directory,
        summary_path=summary_path,
        details_path=details_path,
        total_queries=len(queries),
        successful_queries=successful_queries,
        failed_queries=failed_queries,
        metrics=metrics,
    )
