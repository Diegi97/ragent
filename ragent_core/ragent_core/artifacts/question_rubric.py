from enum import StrEnum
from typing import Annotated, Any, NewType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

DataSourceName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

DEFAULT_RUBRIC_DATASET_ID = "diegi97/ragent-rubrics"


class EvolutionStrategy(StrEnum):
    MULTI_HOP = "Gated Multi-Hop Chains"
    CONDITIONAL = "Conditional Resolution"
    CROSS_ENTITY = "Cross-Entity Coupling"
    CANDIDATE_SPACE = "Candidate-Space Inflation"
    NEAR_TWINS = "Near-Twin Collisions"
    ALIASES = "Alias & Identity Disambiguation"
    COMPARISON = "Dimensional Comparison"
    ABSENCE = "Absence Verification"


LegacyEvolutionStrategy = NewType("LegacyEvolutionStrategy", str)
StrategyLabel = EvolutionStrategy | LegacyEvolutionStrategy
SupportingDocumentId = Annotated[int, Field(strict=True, ge=0)]


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion: str = Field(min_length=1)
    doc_ids: list[SupportingDocumentId] = Field(min_length=1)

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
    evolution_strategies: list[StrategyLabel] = Field(default_factory=list)
    question: str = Field(min_length=1)
    rubric: list[RubricCriterion] = Field(min_length=1)
    doc_ids: list[SupportingDocumentId] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_question_type(cls, value: Any) -> Any:
        if isinstance(value, dict) and "question_type" in value:
            value = dict(value)
            value.pop("question_type")
        return value

    @field_validator("entity", "question")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("evolution_strategies", mode="before")
    @classmethod
    def validate_evolution_strategies(cls, value: Any) -> list[StrategyLabel]:
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError("evolution strategies must be a list of nonblank names")
        normalized = []
        for item in value:
            label = item.strip()
            try:
                normalized.append(EvolutionStrategy(label))
            except ValueError:
                normalized.append(LegacyEvolutionStrategy(label))
        return normalized

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


class QuestionRubricDatasetRecord(QuestionRubricRecord):
    data_source: DataSourceName


class RubricPreparationSource(BaseModel):
    model_config = ConfigDict(extra="allow")

    data_source: DataSourceName


class QuestionRubricDatasetMetadata(BaseModel):
    """Local-dataset source contract shared by generation and evaluation.

    Pipeline diagnostics and preparation settings are retained as extensions;
    evaluation only depends on the authoritative source identity.
    """

    model_config = ConfigDict(extra="allow")

    prepare_config: RubricPreparationSource
