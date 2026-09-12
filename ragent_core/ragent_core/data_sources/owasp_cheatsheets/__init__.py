import logging

from ragent_core.data_sources import (
    DataSourceSpec,
    normalize_source_dataset,
)

from .prepare import corpus

logger = logging.getLogger(__name__)


def load_data_source() -> DataSourceSpec:
    logger.info(
        "Loading and normalizing OWASP Cheat Sheet Series dataset for QA generation"
    )

    dataset = corpus.load()

    dataset = normalize_source_dataset(dataset)
    description = (
        "The OWASP Cheat Sheet Series provides simple good practice guides for application "
        "developers and defenders to follow. Rather than focusing on detailed best practices "
        "that are impractical for many developers and applications, the cheat sheets aim to "
        "deliver guidance that most teams can actually implement. Documents are markdown "
        "cheat sheets from the OWASP CheatSheetSeries repository."
    )
    return DataSourceSpec(
        dataset=dataset, name="owasp_cheatsheets", description=description
    )
