from collections.abc import Sequence

from data_pipelines.artifacts.ranking import normalize_id
from data_pipelines.pipelines.retrieval_evaluation.contracts import ResultDetail
from ragent_core.retrievers.document import RetrievalResult


def deduplicate_results(results: Sequence[RetrievalResult]) -> list[RetrievalResult]:
    seen: set[str] = set()
    deduplicated: list[RetrievalResult] = []
    for result in results:
        normalized = normalize_id(result.id)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(result)
    return deduplicated


def result_details(results: Sequence[RetrievalResult]) -> list[ResultDetail]:
    return [
        {
            "rank": rank,
            "id": result.id,
            "document_id": result.parent_document_id,
            "score": result.score,
            "title": result.title,
        }
        for rank, result in enumerate(results, start=1)
    ]
