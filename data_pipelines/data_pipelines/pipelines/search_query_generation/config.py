from pathlib import Path

from pydantic import BaseModel, Field, field_validator

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
REPOSITORY_ROOT = PROJECT_ROOT.parent

GENERATOR_MODEL = "accounts/fireworks/models/deepseek-v4-flash-0731"


class RetrievalQueriesConfig(BaseModel):
    """Runtime parameters shared by the CLI and the Prefect flow."""

    table_name: str = Field(min_length=1)
    output_path: Path = PROJECT_ROOT / "data/search_query_generation/queries.jsonl"
    lancedb_db_uri: Path = REPOSITORY_ROOT / "lancedb"
    num_queries: int = Field(default=10, ge=0)
    hard_negatives_per_query: int = Field(default=10, ge=0)
    round_trip_top_k: int = Field(default=50, ge=1)
    candidate_mining_top_k: int = Field(default=50, ge=1)
    contrastive_candidate_count: int = Field(default=25, ge=0)
    generator_model: str = GENERATOR_MODEL
    retriever_namespace: str = "default"
    seed: int = 42
    llm_concurrency: int = Field(default=4, ge=1)
    retriever_concurrency: int = Field(default=2, ge=1)

    @field_validator("output_path", "lancedb_db_uri", mode="after")
    @classmethod
    def resolve_paths(cls, value: Path) -> Path:
        return value.expanduser().resolve()
