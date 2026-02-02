from __future__ import annotations

import logging
import importlib
from dataclasses import dataclass
from typing import Optional, Any

from datasets import Dataset, load_dataset

from ragent_core.config import HF_TOKEN

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSourceSpec:
    dataset: Dataset
    name: Optional[str] = None
    description: Optional[str] = None


def safe_ds_name(dataset_name):
    return dataset_name.replace("-", "_").replace("/", "_").replace(".", "_")


def filter_by_word_count(
    dataset: Dataset,
    text_column: str = "text",
    min_words: int = 50,
    max_words: int = 8000,
) -> Dataset:
    def _predicate(example: dict) -> bool:
        value = example.get(text_column)
        if value is None:
            return False
        if not isinstance(value, str):
            value = str(value)
        word_count = len(value.split())
        return min_words <= word_count <= max_words

    return dataset.filter(_predicate)


def keep_only_core_columns(dataset: Dataset) -> Dataset:
    """
    Filter dataset to only keep id, title, and text columns.
    This ensures all data sources have consistent schemas.

    Args:
        dataset: Input dataset

    Returns:
        Dataset with only id, title, and text columns
    """
    columns_to_keep = {"id", "title", "text"}
    columns_to_remove = [
        col for col in dataset.column_names if col not in columns_to_keep
    ]
    if columns_to_remove:
        return dataset.remove_columns(columns_to_remove)
    return dataset


def get_data_source_loader(dataset_name):
    module_name = safe_ds_name(dataset_name)
    module = importlib.import_module(f".{module_name}", package=__package__)
    return getattr(module, "load_data_source")


def load_corpus(dataset_id: str) -> tuple[Dataset, Optional[str], Optional[str]]:
    """
    Load a corpus dataset by name, using a custom loader if available,
    otherwise falling back to HuggingFace load_dataset.

    Returns:
        A tuple of (Dataset, name, description).
    """
    try:
        loader = get_data_source_loader(dataset_id)
        result = loader()
        spec = normalize_data_source_result(result)
        return spec.dataset, spec.name, spec.description
    except (ModuleNotFoundError, AttributeError, ImportError):
        logger.info(
            f"No preprocessing pipeline found for {dataset_id}, loading from HuggingFace."
        )
        dataset = load_dataset(dataset_id, token=HF_TOKEN)
        # Default to "train" split if multiple splits exist
        if isinstance(dataset, dict):
            if "train" in dataset:
                ds = dataset["train"]
            else:
                ds = dataset[next(iter(dataset.keys()))]
        else:
            ds = dataset
        return ds, None, None


def normalize_data_source_result(result: Any) -> DataSourceSpec:
    if isinstance(result, DataSourceSpec):
        return result
    if isinstance(result, tuple) and len(result) == 2:
        dataset, name, description = result
        if not isinstance(dataset, Dataset):
            raise TypeError(
                "load_data_source must return a Dataset or DataSourceSpec as the first element"
            )
        return DataSourceSpec(dataset=dataset, name=name, description=description)
    if isinstance(result, Dataset):
        return DataSourceSpec(dataset=result, description=None)
    raise TypeError(
        "load_data_source must return a Dataset, (Dataset, description) tuple, or DataSourceSpec"
    )


__all__ = [
    "DataSourceSpec",
    "filter_by_word_count",
    "get_data_source_loader",
    "keep_only_core_columns",
    "load_corpus",
    "normalize_data_source_result",
    "safe_ds_name",
]
