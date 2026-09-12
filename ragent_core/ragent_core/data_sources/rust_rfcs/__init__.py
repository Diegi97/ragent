import logging

from ragent_core.data_sources import (
    DataSourceSpec,
    normalize_source_dataset,
)

from .prepare import corpus

logger = logging.getLogger(__name__)


def load_data_source() -> DataSourceSpec:
    logger.info("Loading and normalizing Rust RFCs dataset for QA generation")

    dataset = corpus.load()

    dataset = normalize_source_dataset(dataset)
    description = (
        "The Rust RFC (request for comments) process provides a consistent, controlled path "
        "for substantial changes to Rust so stakeholders can align on the project's direction. "
        "Non-substantial changes like many bug fixes and documentation improvements can be "
        "handled through the normal GitHub pull request workflow. The corpus includes markdown "
        "RFCs from the rust-lang/rfcs repository."
    )
    return DataSourceSpec(dataset=dataset, name="rust_rfcs", description=description)
