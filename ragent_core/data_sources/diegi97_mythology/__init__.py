import logging
from datasets import load_dataset
from ragent_core.data_sources import (
    DataSourceSpec,
    filter_by_word_count,
    keep_only_core_columns,
)

logger = logging.getLogger(__name__)


def load_data_source() -> DataSourceSpec:
    logger.info("Loading and normalizing diegi97/mythology dataset for QA generation")

    dataset = load_dataset("diegi97/mythology", split="train")

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
        "This dataset is a collection of Wikipedia articles focused on mythology topics. "
        "Each document tells the stories of gods, heroes, and mythical creatures from ancient cultures. "
        "It explains the meaning behind myths, describes legendary figures and their adventures, "
        "or analyzes folklore traditions. The text discusses specific mythological narratives from Greek, "
        "Norse, Egyptian, or other traditions, explaining their cultural significance and the tales themselves."
    )
    return DataSourceSpec(
        dataset=dataset, name="diegi97_mythology", description=description
    )
