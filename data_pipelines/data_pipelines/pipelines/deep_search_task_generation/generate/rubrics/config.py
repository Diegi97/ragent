from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from data_pipelines.pipelines.deep_search_task_generation.config import (
    REPOSITORY_ROOT,
)

DEEP_SEARCH_PROJECT = REPOSITORY_ROOT / "environments/ragent_deep_search"


EVALUATION_CONFIG = DEEP_SEARCH_PROJECT / "evaluation.toml"


DEFAULT_PI_MODEL = "accounts/fireworks/models/deepseek-v4-flash-0731"


DEFAULT_SOLVER_MODEL = "deepseek/deepseek-v4-flash-0731"


class PiThinkingLevel(str, Enum):
    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class RubricGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = DEFAULT_PI_MODEL
    solver_model: str = DEFAULT_SOLVER_MODEL
    thinking: PiThinkingLevel | None = None
    num_question_rubrics: int = Field(default=10, ge=0)
    pi_concurrency: int = Field(default=10, ge=1)
    max_attempts: int = Field(default=4, ge=1)
    random_entities: bool = False
    seed: int = 0

    @field_validator("model", "solver_model")
    @classmethod
    def require_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model must not be blank")
        return value

    @field_validator("thinking", mode="before")
    @classmethod
    def normalize_thinking(cls, value: str | None) -> str | None:
        return value.strip().lower() if value is not None else None
