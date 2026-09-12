from enum import StrEnum
from typing import Any, TypedDict

from ragent_core.retrievers.document import DocumentId


class QueryEvaluationStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class RankMetrics(TypedDict):
    precision: float
    recall: float
    hit_rate: float
    reciprocal_rank: float
    average_precision: float
    ndcg: float


class AggregateCutoffMetrics(TypedDict):
    precision: float | None
    recall: float | None
    hit_rate: float | None
    mrr: float | None
    map: float | None
    ndcg: float | None


class RankStatistics(TypedDict):
    mean_hit_rank: float | None
    median_hit_rank: float | None
    hits: int
    misses: int


class AggregateMetrics(TypedDict):
    query_count: int
    cutoffs: dict[str, AggregateCutoffMetrics]
    rank_statistics: RankStatistics


class EvaluationMetrics(TypedDict):
    chunk: AggregateMetrics
    document: AggregateMetrics


class QueryMetrics(TypedDict):
    chunk: dict[str, RankMetrics]
    document: dict[str, RankMetrics]


class ResultDetail(TypedDict):
    rank: int
    id: DocumentId
    document_id: DocumentId | None
    score: float | None
    title: str


class QueryError(TypedDict):
    type: str
    message: str


class QueryEvaluationDetail(TypedDict):
    query_index: int
    source_line_number: int
    object_id: str | None
    query: str
    positive_chunk_id: DocumentId
    positive_document_id: DocumentId
    status: QueryEvaluationStatus
    error: QueryError | None
    latency_ms: float
    chunk_rank: int | None
    document_rank: int | None
    metrics: QueryMetrics | None
    retrieved_results: list[ResultDetail]


class LatencySummary(TypedDict):
    total_ms: float
    mean_query_ms: float | None
    p50_query_ms: float | None
    p95_query_ms: float | None
    min_query_ms: float | None
    max_query_ms: float | None


class EvaluationLatency(LatencySummary):
    retriever_load_ms: float


class EvaluationCounts(TypedDict):
    total_queries: int
    successful_queries: int
    failed_queries: int
    coverage: float


class EvaluationArtifact(TypedDict):
    schema_version: int
    created_at: str
    dataset: dict[str, Any]
    resolved_config: dict[str, Any]
    output_directory: str
    counts: EvaluationCounts
    latency: EvaluationLatency
    metrics: EvaluationMetrics
