from pathlib import Path

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.models import (
    RubricFinalizePaths,
)
from data_pipelines.pipelines.deep_search_task_generation.project import utc_timestamp


def initialize_rubric_finalize_output(
    prepare_directory: Path, run_id: str
) -> RubricFinalizePaths:
    directory = prepare_directory / f"rubric_finalize_{utc_timestamp()}_{run_id[:8]}"
    directory.mkdir(parents=False, exist_ok=False)
    workspace_directory = directory / "workspace"
    workspace_directory.mkdir(exist_ok=False)
    sessions_directory = directory / "sessions"
    sessions_directory.mkdir(exist_ok=False)
    paths = RubricFinalizePaths(
        directory=directory,
        workspace_directory=workspace_directory,
        outputs_directory=workspace_directory / "outputs",
        sessions_directory=sessions_directory,
        entity_facts=directory / "entity_facts.jsonl",
        question_rubrics=directory / "question_rubrics.jsonl",
        failures=directory / "failures.jsonl",
        metadata=directory / "metadata.json",
        lock=directory / ".records.lock",
    )
    for path in (paths.entity_facts, paths.question_rubrics, paths.failures):
        path.touch(exist_ok=False)
    return paths
