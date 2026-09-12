import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from ragent_core.artifacts.question_rubric import (
    QuestionRubricDatasetMetadata,
    QuestionRubricDatasetRecord,
)
from ragent_core.config import HF_TOKEN

METADATA_FILENAME = "metadata.json"


def _local_jsonl_path(dataset_path: str | Path) -> Path | None:
    path = Path(dataset_path).expanduser()
    if path.is_file():
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"Local task dataset must be a JSONL file: {path}")
        return path.resolve()
    if path.suffix.lower() == ".jsonl":
        raise FileNotFoundError(f"Local task dataset not found: {path.resolve()}")
    return None


def _load_local_data_source(dataset_path: Path) -> str:
    metadata_path = dataset_path.with_name(METADATA_FILENAME)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file for local task dataset not found: {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in local task metadata {metadata_path}: {exc.msg}"
        ) from exc
    try:
        contract = QuestionRubricDatasetMetadata.model_validate(metadata)
    except ValueError as exc:
        raise ValueError(f"Invalid local task metadata {metadata_path}: {exc}") from exc
    return contract.prepare_config.data_source


def _iter_local_rows(dataset_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    data_source = _load_local_data_source(dataset_path)
    with dataset_path.open("r", encoding="utf-8") as dataset:
        for line_number, line in enumerate(dataset, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {dataset_path}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL line {line_number} of {dataset_path} must be an object"
                )
            row["data_source"] = data_source
            yield line_number - 1, row


def _iter_hub_rows(
    dataset_id: str,
    split: str,
) -> Iterator[tuple[int, dict[str, Any]]]:
    dataset = load_dataset(
        dataset_id,
        split=split,
        token=HF_TOKEN,
    )
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"Expected split {split!r} of {dataset_id} to load as a Dataset, "
            f"got {type(dataset).__name__}"
        )
    for idx, row in enumerate(dataset):
        yield idx, row


def iter_dataset_rows(
    dataset_path: str | Path,
    split: str,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read one split from a Hugging Face dataset or a local JSONL file"""
    local_path = _local_jsonl_path(dataset_path)
    rows = (
        _iter_local_rows(local_path)
        if local_path is not None
        else _iter_hub_rows(str(dataset_path), split)
    )
    for index, row in rows:
        try:
            record = QuestionRubricDatasetRecord.model_validate(row)
        except ValueError as exc:
            raise ValueError(
                f"Invalid rubric record {index} in {dataset_path}: {exc}"
            ) from exc
        yield index, record.model_dump(mode="json")
