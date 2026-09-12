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
