from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    QuestionRubricRecord,
)
from data_pipelines.pipelines.deep_search_task_generation.models import (
    EntityFactMemoryRecord,
)

DEFAULT_PI_MODEL = "accounts/fireworks/models/deepseek-v4-flash-0731"


class PiThinkingLevel(str, Enum):
    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


@dataclass(frozen=True)
class RubricFinalizePaths:
    directory: Path
    raw_directory: Path
    outputs_directory: Path
    entity_facts: Path
    question_rubrics: Path
    failures: Path
    metadata: Path
    lock: Path


@dataclass(frozen=True)
class FactWorkspace:
    directory: Path
    facts_directory: Path
    outputs_directory: Path
    entity_index: Path
    validator: Path
    allowed_doc_ids: frozenset[int]


@dataclass(frozen=True)
class QuestionRubricAssignment:
    slot: int
    entity_fact: EntityFactMemoryRecord

    @property
    def filename(self) -> str:
        return f"question_rubric_{self.slot:06d}.md"


@dataclass(frozen=True)
class QuestionRubricAttempt:
    assignment: QuestionRubricAssignment
    record: QuestionRubricRecord | None = None
    error: str | None = None
    phoenix_trace_id: str = ""
