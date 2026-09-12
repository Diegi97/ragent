import re
from pathlib import Path
from typing import Any

from ragent_core.artifacts.question_rubric import (
    EvolutionStrategy,
    QuestionRubricRecord,
)

TITLE = "# Question rubric"
ENTITY_PREFIX = "Entity:"
EVOLUTION_PREFIX = "Evolution strategies:"
QUESTION_HEADING = "## Question"
CRITERIA_HEADING = "## Criteria"
DOCS_HEADING = "## Docs"
DOCS_PREFIX = "Docs:"

DOCUMENT_ID_PATTERN = re.compile(r"\d+")


CRITERION_PATTERN = re.compile(r"(\d+)\.\s+(.+)")


EVOLUTION_STRATEGIES = tuple(strategy.value for strategy in EvolutionStrategy)


NO_EVOLUTION_STRATEGIES = "None"


def _nonblank_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_doc_ids(value: str, *, field_name: str) -> list[int]:
    raw_values = [item.strip() for item in value.split(",")]
    if not value.strip() or any(not item for item in raw_values):
        raise ValueError(f"{field_name} must be a comma-separated list of integers")
    if any(not DOCUMENT_ID_PATTERN.fullmatch(item) for item in raw_values):
        raise ValueError(f"{field_name} contains an invalid document ID")
    return [int(item) for item in raw_values]


def _required_prefixed_value(line: str, prefix: str) -> str:
    if not line.startswith(prefix):
        raise ValueError(f"expected {prefix!r}, received {line!r}")
    value = line.removeprefix(prefix).strip()
    if not value:
        raise ValueError(f"{prefix.rstrip(':')} must not be blank")
    return value


def _parse_evolution_strategies(value: str) -> list[str]:
    if value.casefold() == NO_EVOLUTION_STRATEGIES.casefold():
        return []
    strategies = [item.strip() for item in value.split(",")]
    if any(not item for item in strategies):
        raise ValueError(
            "Evolution strategies must be 'None' or a comma-separated list"
        )
    return [EvolutionStrategy(strategy).value for strategy in strategies]


def parse_question_rubric_markdown(path: Path) -> dict[str, Any]:
    lines = _nonblank_lines(path.read_text(encoding="utf-8"))
    if len(lines) < 9:
        raise ValueError("question-rubric Markdown is incomplete")
    if lines[0] != TITLE:
        raise ValueError("first line must be '# Question rubric'")

    entity = _required_prefixed_value(lines[1], ENTITY_PREFIX)
    evolution_strategies = _parse_evolution_strategies(
        _required_prefixed_value(lines[2], EVOLUTION_PREFIX)
    )
    if lines[3] != QUESTION_HEADING:
        raise ValueError("expected '## Question' after Evolution strategies")
    question = lines[4]
    if lines[5] != CRITERIA_HEADING:
        raise ValueError("expected '## Criteria' after the one-line question")

    try:
        docs_heading_index = lines.index(DOCS_HEADING, 6)
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
                "doc_ids": parse_doc_ids(
                    _required_prefixed_value(docs_line, DOCS_PREFIX),
                    field_name=f"criterion {expected_number} Docs",
                ),
            }
        )

    if docs_heading_index + 2 != len(lines):
        raise ValueError("'## Docs' must contain exactly one comma-separated line")
    return {
        "entity": entity,
        "evolution_strategies": evolution_strategies,
        "question": question,
        "rubric": rubric,
        "doc_ids": parse_doc_ids(
            lines[docs_heading_index + 1],
            field_name="top-level Docs",
        ),
    }


def render_markdown(record: QuestionRubricRecord) -> str:
    strategies = ", ".join(record.evolution_strategies) or NO_EVOLUTION_STRATEGIES
    lines = [
        TITLE,
        f"{ENTITY_PREFIX} {record.entity}",
        f"{EVOLUTION_PREFIX} {strategies}",
        "",
        QUESTION_HEADING,
        record.question,
        "",
        CRITERIA_HEADING,
    ]
    for index, criterion in enumerate(record.rubric, start=1):
        lines.extend(
            [
                f"{index}. {criterion.criterion}",
                f"{DOCS_PREFIX} {','.join(map(str, criterion.doc_ids))}",
            ]
        )
    lines.extend(["", DOCS_HEADING, ",".join(map(str, record.doc_ids))])
    return "\n".join(lines)


def example_markdown() -> str:
    return render_markdown(
        QuestionRubricRecord.model_validate(
            {
                "entity": "the assigned entity name exactly as provided",
                "question": "One self-contained question on one line.",
                "rubric": [
                    {
                        "criterion": "One self-contained, binary-checkable criterion on one line.",
                        "doc_ids": [123],
                    },
                    {
                        "criterion": "A second distinct, binary-checkable criterion on one line.",
                        "doc_ids": [456, 789],
                    },
                ],
                "doc_ids": [123, 456, 789],
            }
        )
    )
