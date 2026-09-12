import logging

from datasets import load_dataset

from ragent_core.data_sources import (
    DataSourceSpec,
    normalize_source_dataset,
)

logger = logging.getLogger(__name__)


def load_data_source() -> DataSourceSpec:
    logger.info(
        "Loading and normalizing diegi97/space_exploration dataset for QA generation"
    )

    dataset = load_dataset("diegi97/space_exploration", split="train")

    dataset = normalize_source_dataset(dataset, id_column="id")
    description = (
        "This dataset is a collection of Wikipedia articles focused on space exploration topics. "
        "Each document explains the history, technology, and science behind space missions, "
        "including how rockets work, challenges of human spaceflight, missions to the Moon or Mars, "
        "spacecraft design, discoveries by space agencies like NASA, orbital mechanics, propulsion systems, "
        "life support, and astronaut experiences during missions."
    )
    return DataSourceSpec(
        dataset=dataset, name="diegi97_space_exploration", description=description
    )
