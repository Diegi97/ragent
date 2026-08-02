import math
import statistics
from collections.abc import Iterable, Sequence
from typing import Any


def normalize_id(value: Any) -> str:
    """Normalize numeric and string identifiers using the retriever convention."""
    return str(value)


def deduplicate_ids(values: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    deduplicated: list[Any] = []
    for value in values:
        if value is None:
            continue
        normalized = normalize_id(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(value)
    return deduplicated


def target_rank(ranked_ids: Sequence[Any], target_id: Any) -> int | None:
    normalized_target = normalize_id(target_id)
    return next(
        (
            rank
            for rank, candidate_id in enumerate(ranked_ids, start=1)
            if normalize_id(candidate_id) == normalized_target
        ),
        None,
    )


def metrics_for_rank(rank: int | None, cutoffs: Sequence[int]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
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
) -> dict[str, Any]:
    cutoff_metrics: dict[str, Any] = {}
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
