import logging
import os

from datasets import Dataset, load_from_disk

from ragent_core.data_sources import (
    DataSourceSpec,
    filter_by_word_count,
    keep_only_core_columns,
)
from .prepare import prepare_dataset

logger = logging.getLogger(__name__)


def load_data_source() -> DataSourceSpec:
    logger.info("Loading and normalizing Rust RFCs dataset for QA generation")

    dataset_path = os.path.join("data", "rust_rfcs", "dataset")
    if not os.path.exists(dataset_path):
        logger.info(
            "Dataset not found at %s. Running prepare_dataset()...", dataset_path
        )
        prepare_dataset()

    dataset = load_from_disk(dataset_path)
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"Expected Dataset but loaded {type(dataset)} from {dataset_path}"
        )

    def _formatter(example, index):
        text = example.get("text")
        title = example.get("title")

        if title and text:
            if not text.lstrip().startswith(f"# {title}"):
                merged_text = f"# {title}\n\n{text}"
            else:
                merged_text = text
        else:
            merged_text = text or title or ""

        return {
            "id": index,
            "title": title,
            "text": merged_text,
        }

    dataset = dataset.map(_formatter, with_indices=True)
    dataset = filter_by_word_count(dataset)
    dataset = keep_only_core_columns(dataset)
    description = (
        "The Rust RFC (request for comments) process provides a consistent, controlled path "
        "for substantial changes to Rust so stakeholders can align on the project's direction. "
        "Non-substantial changes like many bug fixes and documentation improvements can be "
        "handled through the normal GitHub pull request workflow. The corpus includes markdown "
        "RFCs from the rust-lang/rfcs repository."
    )
    return DataSourceSpec(dataset=dataset, name="rust_rfcs", description=description)
