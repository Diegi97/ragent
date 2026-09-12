import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from ragent_core.artifacts.question_rubric import SupportingDocumentId
from ragent_core.judges.criteria import Verdict

EVALUATION_CONFIG_ENV = "RAGENT_EVALUATION_CONFIG"
DATA_SOURCE_ENV = "RAGENT_DATA_SOURCE"
AUDITS_DIRECTORY_ENV = "RAGENT_AUDITS_DIRECTORY"
SOLVER_MODEL_ENV = "RAGENT_SOLVER_MODEL"
PYTHON_EXECUTABLE_ENV = "RAGENT_PYTHON_EXECUTABLE"
AUDITS_DIRECTORY_NAME = ".difficulty_checks"


class AuditExecutionError(RuntimeError):
    """The probe or solver could not complete for the current candidate."""


@dataclass(frozen=True)
class AuditPaths:
    directory: Path
    candidate_name: str

    @property
    def retrieval(self) -> Path:
        return self.directory / f"{self.candidate_name}.retrieval.json"

    @property
    def solver(self) -> Path:
        return self.directory / f"{self.candidate_name}.solver.json"


class AuditState(BaseModel):
    ok: StrictBool
    candidate_sha256: str = Field(min_length=1)


class CompletedAudit(AuditState):
    question: str = Field(min_length=1)


class RetrievalAudit(CompletedAudit):
    model_config = ConfigDict(validate_by_name=True, serialize_by_alias=True)

    supporting_doc_ids: list[SupportingDocumentId]
    retrieved_doc_ids: list[SupportingDocumentId] = Field(default_factory=list)
    missing_doc_ids: list[SupportingDocumentId] = Field(default_factory=list)
    # Preserve the original JSON key while retrieval depth is owned by SEARCH_TOP_K.
    all_supporting_docs_retrieved: StrictBool = Field(
        default=False, alias="all_supporting_docs_in_top_10"
    )
    too_easy: StrictBool = False
    probe_passed: StrictBool


class AuditVerdict(BaseModel):
    verdict: Verdict | None = None
    reason: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def parse_recorded_verdict(cls, value: object) -> Verdict | None:
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise ValueError("audit verdict must be a string or absent")
        return Verdict.from_teacher(value)

    @field_serializer("verdict")
    def serialize_verdict(self, value: Verdict | None) -> str:
        # Historical audits use an empty string when the trace has no verdict.
        return value.value if value is not None else ""


class CriterionJudgment(AuditVerdict):
    id: str = Field(min_length=1)
    criterion: str = ""
    doc_ids: list[SupportingDocumentId] = Field(default_factory=list)
    passed: StrictBool


class SolverAudit(CompletedAudit):
    answer: str = ""
    cited_doc_ids: list[SupportingDocumentId] = Field(default_factory=list)
    judgments: list[CriterionJudgment] = Field(default_factory=list)
    criteria_passed: Annotated[StrictInt, Field(ge=0)]
    criteria_total: Annotated[StrictInt, Field(ge=1)]
    percent_passed: float = Field(ge=0, le=100, allow_inf_nan=False)

    @field_validator("percent_passed", mode="before")
    @classmethod
    def require_numeric_percentage(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("solver audit percent_passed must be numeric")
        return value

    @model_validator(mode="after")
    def require_consistent_results(self) -> "SolverAudit":
        if self.criteria_passed > self.criteria_total:
            raise ValueError("solver passed count exceeds total criteria")
        expected_percent = criteria_pass_percentage(
            self.criteria_passed, self.criteria_total
        )
        if not math.isclose(
            self.percent_passed, expected_percent, rel_tol=0, abs_tol=0.005
        ):
            raise ValueError("solver percentage disagrees with criterion counts")
        if self.judgments:
            if (
                len(self.judgments) != self.criteria_total
                or len({item.id for item in self.judgments}) != self.criteria_total
            ):
                raise ValueError(
                    "solver judgments must cover each criterion exactly once"
                )
            if sum(item.passed for item in self.judgments) != self.criteria_passed:
                raise ValueError("solver judgment results disagree with passed count")
        return self


@dataclass(frozen=True)
class ValidatedAudits:
    retrieval: RetrievalAudit
    solver: SolverAudit


def criteria_pass_percentage(passed: int, total: int) -> float:
    """Serialize the rubric score with the same precision used to validate audits."""
    return round(100 * passed / total, 2)
