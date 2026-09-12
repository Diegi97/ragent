import logging
import time
from collections.abc import Sequence
from datetime import datetime, timezone

from data_pipelines.artifacts.io import write_json, write_jsonl
from data_pipelines.artifacts.ranking import deduplicate_ids, target_rank
from data_pipelines.pipelines.retrieval_evaluation.config import (
    RetrievalEvaluationConfig,
)
from data_pipelines.pipelines.retrieval_evaluation.contracts import (
    EvaluationArtifact,
    EvaluationMetrics,
    QueryEvaluationDetail,
    QueryEvaluationStatus,
)
from data_pipelines.pipelines.retrieval_evaluation.inputs import (
    load_dataset_context,
    load_query_records,
)
from data_pipelines.pipelines.retrieval_evaluation.metrics import (
    aggregate_metrics,
    latency_summary,
    metrics_for_rank,
)
from data_pipelines.pipelines.retrieval_evaluation.models import (
    DatasetContext,
    EvaluationSummary,
    QueryRecord,
)
from data_pipelines.pipelines.retrieval_evaluation.output import create_output_directory
from data_pipelines.pipelines.retrieval_evaluation.results import (
    deduplicate_results,
    result_details,
)
from ragent_core.retrievers.document import RetrievalResult
from ragent_core.retrievers.retriever import TurbopufferRetriever

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    def __init__(self, config: RetrievalEvaluationConfig, context: DatasetContext):
        self.config = config
        self.context = context
        self.retriever: TurbopufferRetriever

    def evaluate(self, queries: list[QueryRecord]) -> EvaluationSummary:
        load_started = time.perf_counter()
        self.retriever = self._load_retriever()
        retriever_load_ms = (time.perf_counter() - load_started) * 1000.0

        details: list[QueryEvaluationDetail] = []
        chunk_ranks: list[int | None] = []
        document_ranks: list[int | None] = []
        latencies_ms: list[float] = []
        failed_queries = 0
        evaluation_started = time.perf_counter()

        for record in queries:
            query_started = time.perf_counter()
            try:
                raw_results = self._retrieve(record.query)
                results = deduplicate_results(raw_results)
                chunk_ids = [result.id for result in results]
                document_ids = deduplicate_ids(
                    result.parent_document_id for result in results
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
                        "status": QueryEvaluationStatus.SUCCESS,
                        "error": None,
                        "latency_ms": latency_ms,
                        "chunk_rank": chunk_rank,
                        "document_rank": document_rank,
                        "metrics": {
                            "chunk": metrics_for_rank(chunk_rank, self.config.cutoffs),
                            "document": metrics_for_rank(
                                document_rank, self.config.cutoffs
                            ),
                        },
                        "retrieved_results": result_details(results),
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
                        "status": QueryEvaluationStatus.ERROR,
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
        metrics: EvaluationMetrics = {
            "chunk": aggregate_metrics(chunk_ranks, self.config.cutoffs),
            "document": aggregate_metrics(document_ranks, self.config.cutoffs),
        }
        output_directory = create_output_directory(self.config)
        summary_path = output_directory / "summary.json"
        details_path = output_directory / "details.jsonl"
        coverage = successful_queries / len(queries)
        summary_record: EvaluationArtifact = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": self.context.to_provenance(),
            "resolved_config": {
                **self.config.model_dump(mode="json"),
                "table_name": self.context.table_name,
                "logical_namespace": self.context.logical_namespace,
                "embedding_backend": self.config.embedding_backend,
                "reranker_backend": self.config.reranker_backend,
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
                **latency_summary(latencies_ms, evaluation_ms),
            },
            "metrics": metrics,
        }
        write_jsonl(details_path, details)
        write_json(summary_path, summary_record)
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

    def _load_retriever(self) -> TurbopufferRetriever:
        use_embeddings = self.config.mode.needs_embeddings
        return TurbopufferRetriever.load_index(
            namespace=self.context.logical_namespace,
            model_name=self.config.embedding_model,
            device=self.config.device,
            trust_remote_code=self.config.trust_remote_code,
            reranker_model_name=self.config.reranker_model
            if self.config.reranker
            else None,
            rerank_threshold=self.config.reranker_threshold,
            top_rerank=self.config.reranker_candidate_k,
            rerank_batch_size=self.config.reranker_batch_size,
            max_seq_length=self.config.max_seq_length,
            embedding_service_url=(
                self.config.embedding_service_url if use_embeddings else None
            ),
            reranker_service_url=(
                self.config.reranker_service_url if self.config.reranker else None
            ),
            load_embedding_backend=use_embeddings,
        )

    def _retrieve(self, query: str) -> Sequence[RetrievalResult]:
        return self.retriever.retrieve(
            query,
            table_name=self.context.table_name,
            top_k=self.config.top_k,
            retrieval_mode=self.config.mode,
        )


def evaluate_retrieval(config: RetrievalEvaluationConfig) -> EvaluationSummary:
    """Evaluate one Turbopuffer configuration against a generated query artifact."""
    context = load_dataset_context(config.input_directory)
    queries = load_query_records(context.queries_path)
    return RetrievalEvaluator(config, context).evaluate(queries)
