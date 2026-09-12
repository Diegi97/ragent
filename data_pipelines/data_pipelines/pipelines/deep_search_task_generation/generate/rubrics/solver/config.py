import os
from dataclasses import dataclass
from pathlib import Path

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.audit_contract import (
    AUDITS_DIRECTORY_ENV,
    DATA_SOURCE_ENV,
    EVALUATION_CONFIG_ENV,
    SOLVER_MODEL_ENV,
)

SOLVER_EPISODE_TIMEOUT_SECONDS = 10 * 60


SOLVER_EPISODE_MAX_RETRIES = 1


@dataclass(frozen=True)
class SolverSettings:
    evaluation_config: Path
    data_source: str
    audits_directory: Path
    solver_model: str

    @classmethod
    def from_environment(cls) -> "SolverSettings":
        values = {
            name: os.getenv(name)
            for name in (
                EVALUATION_CONFIG_ENV,
                DATA_SOURCE_ENV,
                AUDITS_DIRECTORY_ENV,
                SOLVER_MODEL_ENV,
            )
        }
        missing = [name for name, value in values.items() if not (value or "").strip()]
        if missing:
            raise ValueError(
                "required environment variables are not set: " + ", ".join(missing)
            )

        evaluation_config = (
            Path(values[EVALUATION_CONFIG_ENV] or "").expanduser().resolve()
        )
        audits_directory = (
            Path(values[AUDITS_DIRECTORY_ENV] or "").expanduser().resolve()
        )
        if not evaluation_config.is_file():
            raise FileNotFoundError(
                f"evaluation config does not exist: {evaluation_config}"
            )
        if not audits_directory.is_dir():
            raise FileNotFoundError(
                f"audits directory does not exist: {audits_directory}"
            )
        return cls(
            evaluation_config=evaluation_config,
            data_source=(values[DATA_SOURCE_ENV] or "").strip(),
            audits_directory=audits_directory,
            solver_model=(values[SOLVER_MODEL_ENV] or "").strip(),
        )
