"""Upload generated question-rubric records to a private Hugging Face dataset.

The input batch is reproducibly divided between the dataset's standard
``train`` and ``test`` splits. Existing split assignments are preserved. Pass
``--replace-data`` to replace records from one data source instead of appending
another batch.

Usage:
    uv run --project data_pipelines python \
        data_pipelines/scripts/upload_question_rubrics.py \
        path/to/question_rubrics.jsonl gitlab_handbook
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    List,
    Value,
    concatenate_datasets,
    load_dataset,
)
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import HfApi
from pydantic import ValidationError

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    QuestionRubricRecord,
)
from ragent_core.config import HF_TOKEN

logger = logging.getLogger(__name__)

DEFAULT_HF_DATASET_ID = "diegi97/ragent-rubrics"
DEFAULT_TEST_SIZE = 0.1
SPLIT_SEED = 42
SPLIT_NAMES = ("train", "test")
QUESTION_RUBRIC_FEATURES = Features(
    {
        "entity": Value("string"),
        "evolution_strategies": List(Value("string")),
        "question": Value("string"),
        "rubric": List(
            {
                "criterion": Value("string"),
                "doc_ids": List(Value("int64")),
            }
        ),
        "doc_ids": List(Value("int64")),
        "data_source": Value("string"),
    }
)


def _test_size(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("test size must be a number") from exc
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("test size must be between 0 and 1")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path", type=Path)
    parser.add_argument(
        "data_source",
        help="Value stored in the data_source column for every input record.",
    )
    parser.add_argument(
        "dataset_id",
        nargs="?",
        default=DEFAULT_HF_DATASET_ID,
        help=(
            f"Optional Hugging Face dataset ID. Defaults to {DEFAULT_HF_DATASET_ID}."
        ),
    )
    parser.add_argument(
        "--replace-data",
        action="store_true",
        help=(
            "Remove existing records for this data source from both splits "
            "before adding the input batch."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=_test_size,
        default=DEFAULT_TEST_SIZE,
        help=f"Test split proportion. Defaults to {DEFAULT_TEST_SIZE}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and split the local JSONL without contacting Hugging Face.",
    )
    return parser.parse_args()


def load_question_rubrics(jsonl_path: Path, data_source: str) -> Dataset:
    """Load and validate every JSONL record, then add its data source."""
    jsonl_path = jsonl_path.expanduser().resolve()
    data_source = data_source.strip()
    if not data_source:
        raise ValueError("data source must not be blank")
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Question-rubric JSONL not found: {jsonl_path}")

    records: list[dict[str, Any]] = []
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


def load_remote_dataset(dataset_id: str) -> DatasetDict | None:
    """Load an existing Hub dataset, returning None when it does not exist."""
    try:
        dataset = load_dataset(dataset_id, token=HF_TOKEN)
    except DatasetNotFoundError:
        return None
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected {dataset_id} to load as a DatasetDict, got {type(dataset).__name__}"
        )
    return dataset


def _align_existing_split(dataset: Dataset, features: Features) -> Dataset:
    expected_columns = list(features)
    actual_columns = set(dataset.column_names)
    if "question_type" in actual_columns:
        dataset = dataset.remove_columns("question_type")
        actual_columns = set(dataset.column_names)
    missing_columns = set(expected_columns).difference(actual_columns)
    if missing_columns == {"evolution_strategies"}:
        dataset = dataset.add_column(
            "evolution_strategies",
            [[] for _ in range(len(dataset))],
            feature=features["evolution_strategies"],
        )
        actual_columns = set(dataset.column_names)
        missing_columns = set(expected_columns).difference(actual_columns)
    extra_columns = actual_columns.difference(expected_columns)
    if missing_columns or extra_columns:
        details: list[str] = []
        if missing_columns:
            details.append("missing " + ", ".join(sorted(missing_columns)))
        if extra_columns:
            details.append("unexpected " + ", ".join(sorted(extra_columns)))
        raise ValueError(
            "Existing dataset schema is incompatible: " + "; ".join(details)
        )
    try:
        return dataset.select_columns(expected_columns).cast(features)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Existing dataset schema is incompatible: {exc}") from exc


def merge_datasets(
    existing: DatasetDict | None,
    incoming: DatasetDict,
    *,
    data_source: str,
    replace_data: bool,
) -> DatasetDict:
    """Append incoming splits, optionally removing one source beforehand."""
    if existing is None:
        return incoming

    unexpected_splits = set(existing).difference(SPLIT_NAMES)
    if unexpected_splits:
        raise ValueError(
            "Existing dataset contains unsupported splits: "
            + ", ".join(sorted(unexpected_splits))
        )

    merged = DatasetDict()
    for split_name in SPLIT_NAMES:
        incoming_split = incoming[split_name]
        existing_split = existing.get(split_name)
        if existing_split is None:
            merged[split_name] = incoming_split
            continue

        existing_split = _align_existing_split(
            existing_split,
            incoming_split.features,
        )
        if replace_data:
            existing_split = existing_split.filter(
                lambda existing_source: existing_source != data_source,
                input_columns=["data_source"],
                desc=f"Removing {data_source} from {split_name}",
            )
        merged[split_name] = concatenate_datasets([existing_split, incoming_split])
    return merged


def publish_dataset(dataset: DatasetDict, dataset_id: str, data_source: str) -> None:
    """Create or update a Hub dataset and enforce private visibility."""
    api = HfApi(token=HF_TOKEN)
    api.create_repo(
        dataset_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    api.update_repo_settings(
        dataset_id,
        repo_type="dataset",
        private=True,
    )
    dataset.push_to_hub(
        dataset_id,
        private=True,
        token=HF_TOKEN,
        commit_message=f"Update question rubrics: {data_source}",
    )


def _log_split_summary(dataset: DatasetDict, prefix: str) -> None:
    for split_name in SPLIT_NAMES:
        split = dataset[split_name]
        logger.info("%s %s: %d rows", prefix, split_name, len(split))


def main() -> None:
    args = _parse_args()
    data_source = args.data_source.strip()
    incoming = split_dataset(
        load_question_rubrics(args.jsonl_path, data_source),
        args.test_size,
    )
    _log_split_summary(incoming, "Incoming")

    if args.dry_run:
        logger.info("Dry run complete; Hugging Face was not contacted.")
        return

    existing = load_remote_dataset(args.dataset_id)
    merged = merge_datasets(
        existing,
        incoming,
        data_source=data_source,
        replace_data=args.replace_data,
    )
    _log_split_summary(merged, "Final")
    publish_dataset(merged, args.dataset_id, data_source)
    logger.info(
        "Uploaded private dataset to https://huggingface.co/datasets/%s",
        args.dataset_id,
    )


if __name__ == "__main__":
    main()
