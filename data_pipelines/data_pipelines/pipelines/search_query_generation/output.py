from datetime import datetime
from pathlib import Path
from typing import Any

from data_pipelines.artifacts.append import append_jsonl
from data_pipelines.artifacts.inputs import count_jsonl
from data_pipelines.artifacts.io import write_json
from data_pipelines.artifacts.paths import output_slug
from data_pipelines.pipelines.search_query_generation.config import (
    RetrievalQueriesConfig,
)
from data_pipelines.pipelines.search_query_generation.metadata import QueryRunMetadata
from data_pipelines.pipelines.search_query_generation.models import (
    ObjectRunSummary,
    OutputPaths,
    RetrievalQuery,
)
from data_pipelines.timestamps import utc_timestamp


def run_output_directory(
    config: RetrievalQueriesConfig,
    run_id: str,
    created_at: datetime | None = None,
) -> Path:
    table_slug = output_slug(config.table_name, fallback="table")
    timestamp = utc_timestamp(created_at=created_at)
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
    append_jsonl(destination, record, paths.lock_path)
    return destination


def write_metadata(
    config: RetrievalQueriesConfig,
    paths: OutputPaths,
    run_id: str,
    batch_timestamp: str,
    phoenix_project: str,
    summaries: list[ObjectRunSummary],
) -> dict[str, Any]:
    trainable_count = count_jsonl(paths.output_path)
    failure_count = count_jsonl(paths.failures_path)
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
    metadata = QueryRunMetadata.model_validate(metadata).model_dump(
        mode="json", exclude_unset=True
    )
    write_json(paths.metadata_path, metadata)
    return metadata
