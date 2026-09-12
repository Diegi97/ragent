import math
import statistics
from collections.abc import Sequence

from data_pipelines.pipelines.retrieval_evaluation.contracts import (
    AggregateCutoffMetrics,
    AggregateMetrics,
    LatencySummary,
    RankMetrics,
)


def metrics_for_rank(
    rank: int | None, cutoffs: Sequence[int]
) -> dict[str, RankMetrics]:
    metrics: dict[str, RankMetrics] = {}
    for cutoff in cutoffs:
        hit = rank is not None and rank <= cutoff
        reciprocal_rank = 1.0 / rank if hit and rank is not None else 0.0
        metrics[str(cutoff)] = {
            "precision": (1.0 / cutoff) if hit else 0.0,
            "recall": 1.0 if hit else 0.0,
            "hit_rate": 1.0 if hit else 0.0,
            "reciprocal_rank": reciprocal_rank,
            "average_precision": reciprocal_rank,
            "ndcg": (1.0 / math.log2(rank + 1) if hit and rank is not None else 0.0),
        }
    return metrics


def aggregate_metrics(
    ranks: Sequence[int | None],
    cutoffs: Sequence[int],
) -> AggregateMetrics:
    cutoff_metrics: dict[str, AggregateCutoffMetrics] = {}
    for cutoff in cutoffs:
        contributions = [
            metrics_for_rank(rank, (cutoff,))[str(cutoff)] for rank in ranks
        ]
        if contributions:
            cutoff_metrics[str(cutoff)] = {
                "precision": statistics.fmean(
                    contribution["precision"] for contribution in contributions
                ),
                "recall": statistics.fmean(
                    contribution["recall"] for contribution in contributions
                ),
                "hit_rate": statistics.fmean(
                    contribution["hit_rate"] for contribution in contributions
                ),
                "mrr": statistics.fmean(
                    contribution["reciprocal_rank"] for contribution in contributions
                ),
                "map": statistics.fmean(
                    contribution["average_precision"] for contribution in contributions
                ),
                "ndcg": statistics.fmean(
                    contribution["ndcg"] for contribution in contributions
                ),
            }
        else:
            cutoff_metrics[str(cutoff)] = {
                "precision": None,
                "recall": None,
                "hit_rate": None,
                "mrr": None,
                "map": None,
                "ndcg": None,
            }

    hit_ranks = [rank for rank in ranks if rank is not None]
    return {
        "query_count": len(ranks),
        "cutoffs": cutoff_metrics,
        "rank_statistics": {
            "mean_hit_rank": statistics.fmean(hit_ranks) if hit_ranks else None,
            "median_hit_rank": statistics.median(hit_ranks) if hit_ranks else None,
            "hits": len(hit_ranks),
            "misses": len(ranks) - len(hit_ranks),
        },
    }


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


def latency_summary(latencies_ms: Sequence[float], total_ms: float) -> LatencySummary:
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
