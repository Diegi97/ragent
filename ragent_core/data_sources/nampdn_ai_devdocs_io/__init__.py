import logging
from datasets import Dataset, load_dataset
from ragent_core.data_sources import (
    DataSourceSpec,
    filter_by_word_count,
    keep_only_core_columns,
)
from .utils import (
    compute_latest_version_map_from_dataset,
    is_latest_label,
)

logger = logging.getLogger(__name__)

# Arbitrary list of technologies I wanted to keep
ALLOWED_LANGUAGES = {
    "terraform",
    "scikit_learn",
    "godot~3.5",
    "rails~7.0",
    "postgresql~15",
    "haskell~9",
    "ruby~3.2",
    "pytorch",
    "c",
    "redis",
    "sqlite",
    "latex",
    "http",
    "docker",
    "deno",
    "html",
    "django~4.2",
    "ocaml",
    "git",
    "go",
    "astro",
    "bash",
    "bootstrap~5",
    "gnuplot",
    "kubernetes",
    "typescript",
    "react",
    "node",
    "homebrew",
    "django_rest_framework",
    "trio",
    "requests",
    "nix",
    "markdown",
    "vite",
}


def filter_by_language(dataset: Dataset, language_column: str = "language") -> Dataset:
    def _predicate(example):
        return example[language_column] in ALLOWED_LANGUAGES

    return dataset.filter(_predicate)


def keep_latest_versions(
    dataset: Dataset, language_column: str = "language"
) -> Dataset:
    base_to_max = compute_latest_version_map_from_dataset(
        dataset, language_column=language_column
    )

    def _predicate(example):
        return is_latest_label(str(example[language_column]), base_to_max)

    return dataset.filter(_predicate)


def add_incrementing_id(
    dataset: Dataset, id_column: str = "id", id_start: int = 0
) -> Dataset:
    ids = list(range(id_start, id_start + len(dataset)))
    return dataset.add_column(id_column, ids)


def load_data_source() -> DataSourceSpec:
    logger.info(
        "Loading and normalizing nampdn-ai/devdocs.io dataset for QA generation"
    )

    dataset = load_dataset("nampdn-ai/devdocs.io", split="train")

    dataset = filter_by_language(dataset)
    dataset = keep_latest_versions(dataset)
    dataset = filter_by_word_count(dataset)
    dataset = add_incrementing_id(dataset)
    dataset = keep_only_core_columns(dataset)

    description = (
        "This dataset contains technical documentation from devdocs.io. Documents are "
        "API references, guides, and tutorials for software libraries and frameworks, "
        "often versioned and highly technical with code snippets and definitions."
    )
    return DataSourceSpec(
        dataset=dataset, name="nampdn_ai_devdocs_io", description=description
    )
