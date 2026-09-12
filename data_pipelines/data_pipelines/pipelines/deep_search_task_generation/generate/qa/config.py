from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_QA_MODEL_ID = "accounts/fireworks/models/kimi-k3"


class QAGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qa_pairs_per_entity: int = Field(default=4, ge=0)
    qa_model_id: str = DEFAULT_QA_MODEL_ID
    complex_pair_ratio: float = Field(default=0.7, ge=0, le=1)
    max_qa_generation_attempts: int = Field(default=4, ge=1)
    llm_concurrency: int = Field(default=25, ge=1)
    download_timeout: float = Field(default=600.0, gt=0, allow_inf_nan=False)

    @field_validator("qa_model_id")
    @classmethod
    def require_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("qa_model_id must not be blank")
        return value
