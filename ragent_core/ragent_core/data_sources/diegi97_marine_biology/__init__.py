import logging

from datasets import load_dataset

from ragent_core.data_sources import (
    DataSourceSpec,
    normalize_source_dataset,
)

logger = logging.getLogger(__name__)


def load_data_source() -> DataSourceSpec:
    logger.info(
        "Loading and normalizing diegi97/marine_biology dataset for QA generation"
    )

    dataset = load_dataset("diegi97/marine_biology", split="train")

    dataset = normalize_source_dataset(dataset, id_column="id")
    description = (
        "This dataset is a collection of Wikipedia articles focused on marine biology topics. "
        "Each document covers underwater ecosystems, the biology and behavior of ocean creatures, "
        "or marine research, including topics like fish, whales, sharks, coral reefs, deep sea habitats, "
        "marine food chains, ocean conservation, and the physiology of sea animals."
    )
    return DataSourceSpec(
        dataset=dataset, name="diegi97_marine_biology", description=description
    )
