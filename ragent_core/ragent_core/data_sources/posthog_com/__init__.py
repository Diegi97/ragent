import logging

from ragent_core.data_sources import DataSourceSpec, filter_by_word_count
from ragent_core.data_sources.records import source_record

from .prepare import corpus

logger = logging.getLogger(__name__)


def load_data_source() -> DataSourceSpec:
    logger.info("Loading and normalizing PostHog docs dataset for QA generation")

    dataset = corpus.load()

    def _formatter(example, index):
        text = example.get("text")
        title = example.get("title")

        record = source_record(index, title, text)
        record["path"] = example.get("path")
        return record

    dataset = dataset.map(_formatter, with_indices=True)
    dataset = filter_by_word_count(dataset)
    description = (
        "This dataset contains PostHog documentation and site content written in MDX. "
        "Documents include product guides, tutorials, and marketing pages with headings, "
        "examples, and structured sections."
    )
    return DataSourceSpec(dataset=dataset, name="posthog_com", description=description)
