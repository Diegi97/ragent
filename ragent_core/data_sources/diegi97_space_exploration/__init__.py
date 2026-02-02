import logging
from datasets import load_dataset
from ragent_core.data_sources import (
    DataSourceSpec,
    filter_by_word_count,
    keep_only_core_columns,
)

logger = logging.getLogger(__name__)


def load_data_source() -> DataSourceSpec:
    logger.info(
        "Loading and normalizing diegi97/space_exploration dataset for QA generation"
    )

    dataset = load_dataset("diegi97/space_exploration", split="train")

    def _formatter(example, index):
        text = example.get("text")
        title = example.get("title")
        if title and text:
            merged_text = f"# {title}\n\n{text}"
        else:
            merged_text = text or title or ""
        return {
            "id": int(example.get("id", index)),
            "title": title,
            "text": merged_text,
        }

    dataset = dataset.map(_formatter, with_indices=True)
    dataset = filter_by_word_count(dataset)
    dataset = keep_only_core_columns(dataset)
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
