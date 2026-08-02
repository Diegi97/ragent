# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pydantic>=2.11,<3",
# ]
# ///

import argparse
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)
INTEGER_PATTERN = re.compile(r"\d+")
CRITERION_PATTERN = re.compile(r"(\d+)\.\s+(.+)")
DOCUMENT_HEADING_PATTERN = re.compile(r"### Documents? (.+)")


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion: str = Field(min_length=1)
    doc_ids: list[StrictInt] = Field(min_length=1)

    @field_validator("criterion")
    @classmethod
    def normalize_criterion(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("criterion must not be blank")
        return value

    @field_validator("doc_ids")
    @classmethod
    def require_unique_doc_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("criterion doc_ids must be unique")
        return value


class QuestionRubricRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: str = Field(min_length=1)
    question_type: str = Field(min_length=1)
    question: str = Field(min_length=1)
    rubric: list[RubricCriterion] = Field(min_length=2)
    doc_ids: list[StrictInt] = Field(min_length=3)

    @field_validator("entity", "question_type", "question")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def validate_rubric(self) -> "QuestionRubricRecord":
        criteria = [" ".join(item.criterion.lower().split()) for item in self.rubric]
        if len(criteria) != len(set(criteria)):
            raise ValueError("rubric criteria must be unique")
        if len(self.doc_ids) != len(set(self.doc_ids)):
            raise ValueError("top-level doc_ids must be unique")
        criterion_doc_ids = {
            doc_id for criterion in self.rubric for doc_id in criterion.doc_ids
        }
        if set(self.doc_ids) != criterion_doc_ids:
            raise ValueError(
                "top-level doc_ids must equal the union of rubric criterion doc_ids"
            )
        return self


def _nonblank_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_doc_ids(value: str, *, field_name: str) -> list[int]:
    raw_values = [item.strip() for item in value.split(",")]
    if not value.strip() or any(not item for item in raw_values):
        raise ValueError(f"{field_name} must be a comma-separated list of integers")
    if any(not INTEGER_PATTERN.fullmatch(item) for item in raw_values):
        raise ValueError(f"{field_name} contains an invalid document ID")
    return [int(item) for item in raw_values]


def _required_prefixed_value(line: str, prefix: str) -> str:
    if not line.startswith(prefix):
        raise ValueError(f"expected {prefix!r}, received {line!r}")
    value = line.removeprefix(prefix).strip()
    if not value:
        raise ValueError(f"{prefix.rstrip(':')} must not be blank")
    return value


def _load_workspace_doc_ids(facts_directory: Path) -> set[int]:
    doc_ids: set[int] = set()
    for path in facts_directory.rglob("*.md"):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = DOCUMENT_HEADING_PATTERN.fullmatch(line.strip())
            if match is None:
                continue
            doc_ids.update(
                _parse_doc_ids(
                    match.group(1),
                    field_name=f"document heading in {path}",
                )
            )
    if not doc_ids:
        raise ValueError(f"no document headings found in {facts_directory}")
    return doc_ids


def parse_question_rubric_markdown(path: Path) -> dict[str, Any]:
    lines = _nonblank_lines(path.read_text(encoding="utf-8"))
    if len(lines) < 10:
        raise ValueError("question-rubric Markdown is incomplete")
    if lines[0] != "# Question rubric":
        raise ValueError("first line must be '# Question rubric'")

    entity = _required_prefixed_value(lines[1], "Entity:")
    question_type = _required_prefixed_value(lines[2], "Type:")
    if lines[3] != "## Question":
        raise ValueError("expected '## Question' after Type")
    question = lines[4]
    if lines[5] != "## Criteria":
        raise ValueError("expected '## Criteria' after the one-line question")

    try:
        docs_heading_index = lines.index("## Docs", 6)
    except ValueError as exc:
        raise ValueError("missing '## Docs' section") from exc
    criterion_lines = lines[6:docs_heading_index]
    if not criterion_lines or len(criterion_lines) % 2:
        raise ValueError(
            "each criterion must use one numbered line followed by one Docs line"
        )

    rubric: list[dict[str, Any]] = []
    for offset in range(0, len(criterion_lines), 2):
        criterion_line = criterion_lines[offset]
        docs_line = criterion_lines[offset + 1]
        match = CRITERION_PATTERN.fullmatch(criterion_line)
        expected_number = offset // 2 + 1
        if match is None or int(match.group(1)) != expected_number:
            raise ValueError(
                f"criterion {expected_number} must start with '{expected_number}. '"
            )
        rubric.append(
            {
                "criterion": match.group(2).strip(),
                "doc_ids": _parse_doc_ids(
                    _required_prefixed_value(docs_line, "Docs:"),
                    field_name=f"criterion {expected_number} Docs",
                ),
            }
        )

    if docs_heading_index + 2 != len(lines):
        raise ValueError("'## Docs' must contain exactly one comma-separated line")
    return {
        "entity": entity,
        "question_type": question_type,
        "question": question,
        "rubric": rubric,
        "doc_ids": _parse_doc_ids(
            lines[docs_heading_index + 1],
            field_name="top-level Docs",
        ),
    }


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a PI-generated question-rubric Markdown file."
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        allowed_doc_ids = _load_workspace_doc_ids(Path.cwd() / "facts")
        validate_question_rubric_file(args.path, allowed_doc_ids=allowed_doc_ids)
    except (OSError, ValueError, ValidationError) as exc:
        logger.error("Invalid question-rubric file: %s", exc)
        return 1
    logger.info("Question-rubric file is valid: %s", args.path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
