import argparse
import logging
from pathlib import Path

from datasets import (
    DatasetDict,
)

from data_pipelines.publishing.question_rubrics import (
    load_remote_dataset,
    publish_dataset,
)
from data_pipelines.publishing.question_rubrics.datasets import (
    DEFAULT_TEST_SIZE,
    SPLIT_NAMES,
    load_question_rubrics,
    merge_datasets,
    same_records,
    split_dataset,
)
from ragent_core.artifacts.question_rubric import (
    DEFAULT_RUBRIC_DATASET_ID,
)

logger = logging.getLogger(__name__)


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
        default=DEFAULT_RUBRIC_DATASET_ID,
        help=(
            f"Optional Hugging Face dataset ID. Defaults to {DEFAULT_RUBRIC_DATASET_ID}."
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
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Test split proportion. Defaults to {DEFAULT_TEST_SIZE}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and split the local JSONL without contacting Hugging Face.",
    )
    return parser.parse_args()


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
    if same_records(existing, merged):
        logger.info("Dataset already contains this batch; no upload is needed.")
        return
    publish_dataset(merged, args.dataset_id, data_source)
    logger.info(
        "Uploaded private dataset to https://huggingface.co/datasets/%s",
        args.dataset_id,
    )
