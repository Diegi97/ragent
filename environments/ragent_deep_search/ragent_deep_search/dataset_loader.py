import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from ragent_core.config import HF_TOKEN

DEFAULT_DATASET_ID = "diegi97/ragent-rubrics"
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
    if not isinstance(metadata, dict):
        raise ValueError(f"Local task metadata must be an object: {metadata_path}")

    prepare_config = metadata.get("prepare_config")
    data_source = (
        prepare_config.get("data_source") if isinstance(prepare_config, dict) else None
    )
    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError(
            "Local task metadata is missing a nonblank "
            f"prepare_config.data_source: {metadata_path}"
        )
    return data_source.strip()


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
    if local_path is not None:
        yield from _iter_local_rows(local_path)
        return
    yield from _iter_hub_rows(str(dataset_path), split)
