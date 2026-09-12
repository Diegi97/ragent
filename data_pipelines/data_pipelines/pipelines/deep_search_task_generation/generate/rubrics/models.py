from dataclasses import dataclass, field
from pathlib import Path

from data_pipelines.pipelines.deep_search_task_generation.facts import (
    EntityFactMemoryRecord,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.audit_contract import (
    SolverAudit,
)
from ragent_core.artifacts.question_rubric import QuestionRubricRecord


@dataclass(frozen=True)
class RubricFinalizePaths:
    directory: Path
    workspace_directory: Path
    outputs_directory: Path
    sessions_directory: Path
    entity_facts: Path
    question_rubrics: Path
    failures: Path
    metadata: Path
    lock: Path

    def to_metadata(self) -> dict[str, str]:
        return {
            "rubric_finalize_run_directory": str(self.directory),
            "workspace_directory": str(self.workspace_directory),
            "outputs_directory": str(self.outputs_directory),
            "sessions_directory": str(self.sessions_directory),
            "entity_facts": str(self.entity_facts),
            "question_rubrics": str(self.question_rubrics),
            "failures": str(self.failures),
        }


@dataclass(frozen=True)
class FactWorkspace:
    directory: Path
    facts_directory: Path
    outputs_directory: Path
    entity_index: Path
    validator: Path
    retrieval_probe: Path
    solver: Path
    audits_directory: Path
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
    solver_audit: SolverAudit | None = None
    error: str | None = None
    infrastructure_error: bool = False
    phoenix_trace_id: str = ""


@dataclass
class RubricGenerationResult:
    accepted: dict[int, QuestionRubricRecord] = field(default_factory=dict)
    errors: dict[int, list[str]] = field(default_factory=dict)
    phoenix_trace_ids: dict[int, str] = field(default_factory=dict)
    solver_audits: dict[int, SolverAudit] = field(default_factory=dict)
