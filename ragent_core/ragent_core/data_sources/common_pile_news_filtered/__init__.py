import logging

from datasets import load_dataset

from ragent_core.data_sources import DataSourceSpec, filter_by_word_count
from ragent_core.data_sources.records import source_record

logger = logging.getLogger(__name__)


def _split_title_body(text: str) -> tuple[str, str]:
    lines = [line for line in text.splitlines()]
    if not lines:
        return "", ""
    title = lines[0].strip()
    body = "\n".join(lines[1:]).strip()
    return title, body


def load_data_source() -> DataSourceSpec:
    logger.info(
        "Loading and normalizing common-pile/news_filtered dataset for QA generation"
    )

    dataset = load_dataset("common-pile/news_filtered", split="train")

    def _formatter(example, index):
        text = example.get("text") or ""
        title, body = _split_title_body(text)
        return source_record(index, title, body)

    dataset = dataset.map(_formatter, with_indices=True)
    dataset = filter_by_word_count(dataset)
    description = (
        "This dataset contains news-style articles with a title line followed by "
        "body text. Documents are written in journalistic prose and cover current events "
        "and general-interest topics."
    )
    return DataSourceSpec(
        dataset=dataset, name="common_pile_news_filtered", description=description
    )
