import logging

from ragent_core.data_sources import (
    DataSourceSpec,
    normalize_source_dataset,
)

from .prepare import corpus

logger = logging.getLogger(__name__)


def load_data_source() -> DataSourceSpec:
    logger.info("Loading and normalizing GitLab Handbook dataset for QA generation")

    # GitLab Handbook is stored locally in data/gitlab_handbook/dataset
    dataset = corpus.load()

    dataset = normalize_source_dataset(dataset)
    description = (
        "The GitLab Handbook is a public company handbook covering GitLab's policies, "
        "processes, roles, workflows, and cultural guidelines. Documents are markdown-style "
        "sections with headings, procedural steps, and policy details."
    )
    return DataSourceSpec(
        dataset=dataset, name="gitlab_handbook", description=description
    )
