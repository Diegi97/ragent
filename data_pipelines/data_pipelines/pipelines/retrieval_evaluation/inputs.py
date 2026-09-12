import json
from pathlib import Path

from data_pipelines.artifacts.retrieval_queries import TrainingRecord
from data_pipelines.pipelines.retrieval_evaluation.models import (
    DatasetContext,
    QueryRecord,
)
from data_pipelines.pipelines.search_query_generation.metadata import QueryRunMetadata


def load_dataset_context(input_directory: Path) -> DatasetContext:
    metadata_path = input_directory / "metadata.json"
    if not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {input_directory}")
    if not metadata_path.is_file():
        raise ValueError(f"Missing metadata.json in {input_directory}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read valid metadata from {metadata_path}: {exc}"
        ) from exc
    try:
        source_metadata = QueryRunMetadata.model_validate(metadata)
    except ValueError as exc:
        raise ValueError(f"Invalid search-query run metadata: {exc}") from exc
    source_config = source_metadata.config
    queries_path = input_directory / source_config.output_path.name
    if not queries_path.is_file():
        raise ValueError(f"Missing {queries_path.name} in {input_directory}")
    return DatasetContext(
        input_directory=input_directory,
        queries_path=queries_path,
        metadata_path=metadata_path,
        table_name=source_config.table_name,
        logical_namespace=source_config.logical_namespace,
        source_metadata=source_metadata,
    )


def load_query_records(path: Path) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
        try:
            record = TrainingRecord.model_validate(raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid training record on {path}:{line_number}: {exc}"
            ) from exc
        records.append(
            QueryRecord(
                index=len(records),
                line_number=line_number,
                query=record.query.strip(),
                positive_chunk_id=record.positive.id,
                positive_document_id=record.positive.document_id,
                metadata=record.metadata,
            )
        )
    if not records:
        raise ValueError(f"No query records were found in {path}.")
    return records
