from pathlib import Path

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation.markdown import (
    parse_question_rubric_markdown,
)
from ragent_core.artifacts.question_rubric import QuestionRubricRecord


def validate_question_rubric_file(
    path: Path,
    *,
    allowed_doc_ids: set[int] | frozenset[int] | None = None,
    expected_entity: str | None = None,
) -> QuestionRubricRecord:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"question-rubric file does not exist: {path}")
    record = QuestionRubricRecord.model_validate(
        parse_question_rubric_markdown(path), strict=True
    )
    if expected_entity is not None and record.entity != expected_entity:
        raise ValueError(
            f"entity must be {expected_entity!r}, received {record.entity!r}"
        )
    if allowed_doc_ids is not None:
        unknown_doc_ids = sorted(set(record.doc_ids).difference(allowed_doc_ids))
        if unknown_doc_ids:
            raise ValueError(
                "question-rubric references document IDs absent from the fact graph: "
                + ", ".join(str(value) for value in unknown_doc_ids)
            )
    return record
