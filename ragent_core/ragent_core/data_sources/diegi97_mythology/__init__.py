import logging

from datasets import load_dataset

from ragent_core.data_sources import (
    DataSourceSpec,
    normalize_source_dataset,
)

logger = logging.getLogger(__name__)


def load_data_source() -> DataSourceSpec:
    logger.info("Loading and normalizing diegi97/mythology dataset for QA generation")

    dataset = load_dataset("diegi97/mythology", split="train")

    dataset = normalize_source_dataset(dataset, id_column="id")
    description = (
        "This dataset is a collection of Wikipedia articles focused on mythology topics. "
        "Each document tells the stories of gods, heroes, and mythical creatures from ancient cultures. "
        "It explains the meaning behind myths, describes legendary figures and their adventures, "
        "or analyzes folklore traditions. The text discusses specific mythological narratives from Greek, "
        "Norse, Egyptian, or other traditions, explaining their cultural significance and the tales themselves."
    )
    return DataSourceSpec(
        dataset=dataset, name="diegi97_mythology", description=description
    )
