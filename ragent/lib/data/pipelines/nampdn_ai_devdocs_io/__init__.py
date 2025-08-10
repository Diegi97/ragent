import logging

from datasets import Dataset

logger = logging.getLogger(__name__)


from ragent.lib.data.pipelines.nampdn_ai_devdocs_io.utils import is_latest_label, \
    compute_latest_version_map_from_dataset


def keep_latest_versions(dataset, language_column = "language"):
    base_to_max = compute_latest_version_map_from_dataset(dataset, language_column=language_column)

    def _predicate(example):
        return is_latest_label(str(example[language_column]), base_to_max)

    return dataset.filter(_predicate)

def filter_by_max_word_count(dataset, text_column = "text", max_words = 15000):
    def _should_keep(example):
        value = example.get(text_column) if isinstance(example, dict) else None
        if value is None:
            return True
        if isinstance(value, str):
            return len(value.split()) <= max_words
        try:
            return len(value) <= max_words
        except Exception:
            return True

    return dataset.filter(_should_keep)

def add_incrementing_id(dataset, id_column = "id", id_start = 0):
    ids = list(range(id_start, id_start + len(dataset)))
    return dataset.add_column(id_column, ids)


def run(dataset: Dataset):
    logger.info(f"Running pipeline over {dataset}")
    for t in [
        keep_latest_versions,
        filter_by_max_word_count,
        add_incrementing_id,
    ]:
        dataset = t(dataset)
    return dataset
