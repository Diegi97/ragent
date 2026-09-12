import json
from pathlib import Path
from typing import Any

from datasets import (
    Dataset,
    DatasetDict,
)
from pydantic import ValidationError

from data_pipelines.publishing.question_rubrics.schema import (
    QUESTION_RUBRIC_FEATURES,
    align_existing_split,
)
from ragent_core.artifacts.question_rubric import (
    QuestionRubricRecord,
)

DEFAULT_TEST_SIZE = 0.1
SPLIT_SEED = 42
SPLIT_NAMES = ("train", "test")


def load_question_rubrics(jsonl_path: Path, data_source: str) -> Dataset:
    """Load and validate every JSONL record, then add its data source."""
    jsonl_path = jsonl_path.expanduser().resolve()
    data_source = data_source.strip()
    if not data_source:
        raise ValueError("data source must not be blank")
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Question-rubric JSONL not found: {jsonl_path}")

    records: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {jsonl_path}: {exc.msg}"
                ) from exc
            try:
                record = QuestionRubricRecord.model_validate(
                    raw_record, strict=True
                ).model_dump()
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid question-rubric record on line {line_number} "
                    f"of {jsonl_path}: {exc}"
                ) from exc
            record["data_source"] = data_source
            if _admit_record(record, seen):
                records.append(record)

    if not records:
        raise ValueError(f"Question-rubric JSONL is empty: {jsonl_path}")
    return Dataset.from_list(records, features=QUESTION_RUBRIC_FEATURES)


def split_dataset(dataset: Dataset, test_size: float) -> DatasetDict:
    """Reproducibly split an input batch into train and test records."""
    if not 0 < test_size < 1:
        raise ValueError("test size must be between 0 and 1")
    if len(dataset) < 2:
        raise ValueError("at least two records are required for a train/test split")
    return dataset.train_test_split(
        test_size=test_size,
        seed=SPLIT_SEED,
    )


def merge_datasets(
    existing: DatasetDict | None,
    incoming: DatasetDict,
    *,
    data_source: str,
    replace_data: bool,
) -> DatasetDict:
    """Merge each question once, retaining existing split assignments on replay."""
    existing = existing if existing is not None else DatasetDict()
    unexpected_splits = (set(existing) | set(incoming)).difference(SPLIT_NAMES)
    if unexpected_splits:
        raise ValueError(
            "Dataset contains unsupported splits: "
            + ", ".join(sorted(unexpected_splits))
        )

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    seen: dict[tuple[str, str], dict[str, Any]] = {}

    def admit(record: dict[str, Any], split_name: str) -> None:
        if _admit_record(record, seen):
            rows[split_name].append(record)

    # If an old exact duplicate leaked across both splits, retain its evaluation assignment.
    for split_name in reversed(SPLIT_NAMES):
        if split_name not in existing:
            continue
        aligned = align_existing_split(existing[split_name], QUESTION_RUBRIC_FEATURES)
        for record in aligned:
            if replace_data and record["data_source"] == data_source:
                continue
            admit(record, split_name)
    for split_name in SPLIT_NAMES:
        if split_name not in incoming:
            continue
        for record in incoming[split_name]:
            if record["data_source"] != data_source:
                raise ValueError(
                    "Incoming records must belong to the selected data source"
                )
            admit(record, split_name)
    return DatasetDict(
        {
            name: Dataset.from_list(records, features=QUESTION_RUBRIC_FEATURES)
            if records
            else Dataset.from_dict(
                {column: [] for column in QUESTION_RUBRIC_FEATURES},
                features=QUESTION_RUBRIC_FEATURES,
            )
            for name, records in rows.items()
        }
    )


def same_records(existing: DatasetDict | None, merged: DatasetDict) -> bool:
    """Avoid a Hub commit when an exact replay leaves all split contents unchanged."""
    return (
        existing is not None
        and set(existing) == set(merged)
        and all(existing[name].to_dict() == merged[name].to_dict() for name in merged)
    )


def _admit_record(
    record: dict[str, Any], seen: dict[tuple[str, str], dict[str, Any]]
) -> bool:
    identity = (record["data_source"], " ".join(record["question"].casefold().split()))
    previous = seen.get(identity)
    if previous is not None:
        if previous != record:
            raise ValueError(
                f"Conflicting records for question {record['question']!r} in source "
                f"{record['data_source']!r}; replace that source explicitly to change its records"
            )
        return False
    seen[identity] = record
    return True
