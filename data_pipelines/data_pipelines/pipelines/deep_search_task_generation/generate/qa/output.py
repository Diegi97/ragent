from pathlib import Path
from typing import Any

from data_pipelines.pipelines.deep_search_task_generation.generate.qa.models import (
    FinalizePaths,
)
from data_pipelines.pipelines.deep_search_task_generation.project import utc_timestamp
from ragent_core.types import QA


def initialize_finalize_output(prepare_directory: Path, run_id: str) -> FinalizePaths:
    directory = prepare_directory / f"finalize_{utc_timestamp()}_{run_id[:8]}"
    directory.mkdir(parents=False, exist_ok=False)
    raw_directory = directory / "raw"
    raw_directory.mkdir(exist_ok=False)
    paths = FinalizePaths(
        directory=directory,
        raw_directory=raw_directory,
        entity_facts=directory / "entity_facts.jsonl",
        qas=directory / "qas.jsonl",
        failures=directory / "failures.jsonl",
        metadata=directory / "metadata.json",
        lock=directory / ".records.lock",
    )
    for path in (paths.entity_facts, paths.qas, paths.failures):
        path.touch(exist_ok=False)
    return paths


def qa_to_dict(qa: QA) -> dict[str, Any]:
    return {
        "question": qa.question,
        "answer": qa.answer,
        "doc_ids": qa.doc_ids,
        "info": qa.info,
    }
