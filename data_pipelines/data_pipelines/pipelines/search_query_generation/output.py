import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

from data_pipelines.pipelines.search_query_generation.config import (
    RetrievalQueriesConfig,
)
from data_pipelines.pipelines.search_query_generation.models import (
    ObjectRunSummary,
    OutputPaths,
    RetrievalQuery,
)


def run_output_directory(
    config: RetrievalQueriesConfig,
    run_id: str,
    created_at: datetime | None = None,
) -> Path:
    table_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", config.table_name).strip("-._")
    table_slug = (table_slug or "table")[:80]
    timestamp = (
        (created_at or datetime.now(timezone.utc))
        .astimezone(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )
    return config.output_path.parent / (
        f"{table_slug}_{timestamp}_{config.num_queries}q_{run_id[:8]}"
    )


def initialize_output(
    config: RetrievalQueriesConfig,
    run_id: str,
    created_at: datetime | None = None,
) -> OutputPaths:
    directory = run_output_directory(config, run_id, created_at)
    directory.mkdir(parents=True, exist_ok=False)
    paths = OutputPaths(
        output_directory=directory,
        output_path=directory / config.output_path.name,
        failures_path=directory / "failures.jsonl",
        metadata_path=directory / "metadata.json",
        lock_path=directory / ".records.lock",
    )
    paths.output_path.touch(exist_ok=False)
    paths.failures_path.touch(exist_ok=False)
    return paths


def append_query_record(
    retrieval_query: RetrievalQuery,
    paths: OutputPaths,
    hard_negatives_per_query: int,
) -> Path:
    destination = (
        paths.output_path if retrieval_query.is_trainable() else paths.failures_path
    )
    record = (
        retrieval_query.to_training_record(hard_negatives_per_query)
        if retrieval_query.is_trainable()
        else retrieval_query.to_failure_record()
    )
    encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    with FileLock(paths.lock_path):
        with destination.open("ab", buffering=0) as fp:
            view = memoryview(encoded)
            while view:
                written = fp.write(view)
                if written is None or written <= 0:
                    raise OSError(f"Failed to append a record to {destination}.")
                view = view[written:]
            fp.flush()
            os.fsync(fp.fileno())
    return destination


def count_jsonl_records(path: Path) -> int:
    with path.open("rb") as fp:
        return sum(1 for line in fp if line.strip())


def write_metadata(
    config: RetrievalQueriesConfig,
    paths: OutputPaths,
    run_id: str,
    batch_timestamp: str,
    phoenix_project: str,
    summaries: list[ObjectRunSummary],
) -> dict[str, Any]:
    trainable_count = count_jsonl_records(paths.output_path)
    failure_count = count_jsonl_records(paths.failures_path)
    crashed = [summary for summary in summaries if summary.crashed]
    metadata = {
        "config": config.model_dump(mode="json"),
        "requested_records": config.num_queries,
        "reserved_records": len(summaries),
        "total_records": trainable_count + failure_count,
        "trainable_records": trainable_count,
        "failure_records": failure_count,
        "crashed_records": len(crashed),
        "crashes": [
            {
                "object_id": summary.object_id,
                "sample_index": summary.sample_index,
                "row_index": summary.row_index,
                "error": summary.error,
            }
            for summary in crashed
        ],
        "prefect_flow_run_id": run_id,
        "batch_timestamp": batch_timestamp,
        "phoenix_project": phoenix_project,
        "output_directory": str(paths.output_directory),
        "output_path": str(paths.output_path),
        "failures_path": str(paths.failures_path),
    }
    temporary_path = paths.metadata_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temporary_path, paths.metadata_path)
    return metadata
