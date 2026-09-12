from pathlib import Path

from data_pipelines.pipelines.deep_search_task_generation.facts import (
    ENTITY_FACTS_FILENAME,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.models import (
    FinalizePaths,
)
from data_pipelines.timestamps import utc_timestamp


def initialize_finalize_output(prepare_directory: Path, run_id: str) -> FinalizePaths:
    directory = prepare_directory / f"finalize_{utc_timestamp()}_{run_id[:8]}"
    directory.mkdir(parents=False, exist_ok=False)
    raw_directory = directory / "raw"
    raw_directory.mkdir(exist_ok=False)
    paths = FinalizePaths(
        directory=directory,
        raw_directory=raw_directory,
        entity_facts=directory / ENTITY_FACTS_FILENAME,
        qas=directory / "qas.jsonl",
        failures=directory / "failures.jsonl",
        metadata=directory / "metadata.json",
        lock=directory / ".records.lock",
    )
    for path in (paths.entity_facts, paths.qas, paths.failures):
        path.touch(exist_ok=False)
    return paths
