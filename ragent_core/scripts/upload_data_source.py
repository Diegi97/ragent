"""Upload one RAGent data source as one Hugging Face dataset split.

Each invocation loads a single module from ``ragent_core.data_sources`` and
pushes only that split.

Usage:
    uv run --project ragent_core python ragent_core/scripts/upload_data_source.py \
        posthog_com
"""

import argparse
import logging
import re

from datasets import Dataset, get_dataset_split_names
from datasets.exceptions import DatasetNotFoundError

from ragent_core.config import HF_TOKEN
from ragent_core.data_sources import (
    get_data_source_loader,
    normalize_data_source_result,
    safe_ds_name,
)

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO_ID = "diegi97/ragent_data_sources"
REQUIRED_COLUMNS = frozenset({"id", "title", "text"})
SPLIT_NAME_RE = re.compile(r"^\w+(?:\.\w+)*$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload one ragent_core data source to Hugging Face as an "
            "independent dataset split."
        )
    )
    parser.add_argument(
        "data_source",
        help=(
            "Data-source module name, for example 'posthog_com' or "
            "'nampdn-ai/devdocs.io'."
        ),
    )
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument(
        "--split",
        default=None,
        help="Hub split name. Defaults to the data source's declared name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the split if it already exists. Other splits are preserved.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repository if the target repository does not exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the source without contacting Hugging Face.",
    )
    return parser.parse_args()


def load_data_source(data_source: str) -> tuple[Dataset, str, str | None]:
    """Load one source and resolve the default split name."""
    logger.info("Loading data source: %s", data_source)
    loader = get_data_source_loader(data_source)
    spec = normalize_data_source_result(loader())
    split_name = spec.name or safe_ds_name(data_source)
    return spec.dataset, split_name, spec.description


def validate_dataset(dataset: Dataset, split_name: str) -> None:
    """Validate the shared schema and Hugging Face split-name constraints."""
    if not SPLIT_NAME_RE.fullmatch(split_name):
        raise ValueError(
            f"Invalid split name {split_name!r}; expected letters, numbers, "
            "underscores, or dot-separated components"
        )

    missing_columns = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Data source is missing required columns: {missing}")

    extra_columns = set(dataset.column_names).difference(REQUIRED_COLUMNS)
    if extra_columns:
        logger.warning(
            "Split '%s' contains additional columns: %s",
            split_name,
            ", ".join(sorted(extra_columns)),
        )


def get_remote_splits(repo_id: str) -> set[str]:
    """Return current default-config splits, or an empty set for a new repo."""
    try:
        return set(
            get_dataset_split_names(
                repo_id,
                config_name="default",
                token=HF_TOKEN,
            )
        )
    except DatasetNotFoundError:
        return set()


def upload_data_source(
    *,
    dataset: Dataset,
    repo_id: str,
    split_name: str,
    overwrite: bool,
    private: bool,
) -> None:
    """Upload one split without rebuilding or replacing unrelated splits."""
    remote_splits = get_remote_splits(repo_id)
    if split_name in remote_splits and not overwrite:
        raise ValueError(
            f"Split {split_name!r} already exists in {repo_id}. "
            "Pass --overwrite to replace only that split."
        )

    action = "Replace" if split_name in remote_splits else "Add"
    logger.info(
        "%s split '%s' (%d documents) in %s",
        action,
        split_name,
        len(dataset),
        repo_id,
    )
    dataset.push_to_hub(
        repo_id,
        split=split_name,
        private=private,
        token=HF_TOKEN,
        commit_message=f"{action} data source split: {split_name}",
    )
    logger.info(
        "Uploaded split '%s' to https://huggingface.co/datasets/%s",
        split_name,
        repo_id,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    dataset, default_split_name, description = load_data_source(args.data_source)
    split_name = args.split or default_split_name
    validate_dataset(dataset, split_name)

    logger.info("Prepared split '%s' with %d documents", split_name, len(dataset))
    if description:
        logger.info("Description: %s", description)
    if args.dry_run:
        logger.info("Dry run complete; nothing was uploaded.")
        return

    upload_data_source(
        dataset=dataset,
        repo_id=args.repo_id,
        split_name=split_name,
        overwrite=args.overwrite,
        private=not args.public,
    )


if __name__ == "__main__":
    main()
