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
    logger.info("Loading and normalizing GitLab Handbook dataset for QA generation")

    # GitLab Handbook is stored locally in data/gitlab_handbook/dataset
    dataset_path = os.path.join("data", "gitlab_handbook", "dataset")
    if not os.path.exists(dataset_path):
        logger.info(
            f"Dataset not found at {dataset_path}. Running prepare_dataset()..."
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
        "The GitLab Handbook is a public company handbook covering GitLab's policies, "
        "processes, roles, workflows, and cultural guidelines. Documents are markdown-style "
        "sections with headings, procedural steps, and policy details."
    )
    return DataSourceSpec(
        dataset=dataset, name="gitlab_handbook", description=description
    )
