import importlib
import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from datasets import Dataset, load_dataset

from ragent_core.config import HF_TOKEN
from ragent_core.data_sources.records import CORE_COLUMNS, source_record
from ragent_core.retrievers.document import Document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSourceSpec:
    dataset: Dataset
    name: Optional[str] = None
    description: Optional[str] = None

    @classmethod
    def from_loader_result(cls, result: Any) -> "DataSourceSpec":
        if isinstance(result, cls):
            return result
        if isinstance(result, Dataset):
            return cls(dataset=result)
        if isinstance(result, tuple) and len(result) == 2:
            dataset, description = result
            if not isinstance(dataset, Dataset) or (
                description is not None and not isinstance(description, str)
            ):
                raise TypeError("Loader tuple must contain (Dataset, description)")
            return cls(dataset=dataset, description=description)
        raise TypeError(
            "load_data_source must return a Dataset, (Dataset, description), or DataSourceSpec"
        )


class DataSourceLoader(Protocol):
    def __call__(self) -> Dataset | tuple[Dataset, str | None] | DataSourceSpec: ...


def safe_ds_name(dataset_name: str) -> str:
    return dataset_name.replace("-", "_").replace("/", "_").replace(".", "_")


def normalize_source_dataset(
    dataset: Dataset, *, id_column: str | None = None
) -> Dataset:
    """Normalize source documents, then apply the shared corpus size/schema policy."""

    def format_record(example: dict[str, Any], index: int) -> dict[str, Any]:
        document_id = int(example.get(id_column, index)) if id_column else index
        return source_record(document_id, example.get("title"), example.get("text"))

    dataset = dataset.map(format_record, with_indices=True)
    return keep_only_core_columns(filter_by_word_count(dataset))


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
    columns_to_keep = CORE_COLUMNS
    columns_to_remove = [
        col for col in dataset.column_names if col not in columns_to_keep
    ]
    if columns_to_remove:
        return dataset.remove_columns(columns_to_remove)
    return dataset


def get_data_source_loader(dataset_name: str) -> DataSourceLoader:
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
    except ModuleNotFoundError as exc:
        expected_module = f"{__package__}.{safe_ds_name(dataset_id)}"
        if exc.name != expected_module:
            raise
        logger.info(
            "No preprocessing pipeline found for %s, loading from HuggingFace",
            dataset_id,
        )
        dataset = load_dataset(dataset_id, token=HF_TOKEN)
        if isinstance(dataset, dict):
            dataset = (
                dataset["train"] if "train" in dataset else dataset[next(iter(dataset))]
            )
        spec = DataSourceSpec.from_loader_result(dataset)
    else:
        spec = DataSourceSpec.from_loader_result(loader())

    if "id" not in spec.dataset.column_names:
        raise ValueError(f"Corpus {dataset_id!r} must contain an id column")

    def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        document = Document.from_dict(record)
        return {"id": document.id, "title": document.title, "text": document.content}

    dataset = spec.dataset.map(normalize_record)
    return dataset, spec.name, spec.description


__all__ = [
    "DataSourceSpec",
    "filter_by_word_count",
    "get_data_source_loader",
    "keep_only_core_columns",
    "load_corpus",
    "normalize_source_dataset",
    "safe_ds_name",
]
