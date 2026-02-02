import logging

from datasets import Dataset, load_dataset

from ragent_core.data_sources import (
    DataSourceSpec,
    filter_by_word_count,
    keep_only_core_columns,
)

logger = logging.getLogger(__name__)

TITLE_TOKENS = (" Title: ", " Title:")
STOP_TOKENS = (" Author:", " Status:", " Type:", " Created:", " Python-Version:")


def _extract_title(text: str) -> str:
    first_line = text.splitlines()[0] if text else ""
    if not first_line:
        return ""
    title = ""
    for token in TITLE_TOKENS:
        if token in first_line:
            title = first_line.split(token, 1)[1]
            break
    if not title:
        return first_line.strip()
    for stop in STOP_TOKENS:
        if stop in title:
            title = title.split(stop, 1)[0]
            break
    return title.strip()


def load_data_source() -> DataSourceSpec:
    logger.info(
        "Loading and normalizing common-pile/python_enhancement_proposals_filtered dataset "
        "for QA generation"
    )

    dataset = load_dataset(
        "common-pile/python_enhancement_proposals_filtered", split="train"
    )

    def _formatter(example, index):
        text = example.get("text") or ""
        title = _extract_title(text)
        merged_text = f"# {title}\n\n{text}" if title else text
        return {
            "id": index,
            "title": title or None,
            "text": merged_text,
        }

    dataset = dataset.map(_formatter, with_indices=True)
    dataset = filter_by_word_count(dataset)
    dataset = keep_only_core_columns(dataset)
    description = (
        "This dataset contains Python Enhancement Proposals (PEPs). Documents include "
        "formal proposal headers and long-form technical prose describing changes to the "
        "Python language and standard library."
    )
    return DataSourceSpec(
        dataset=dataset, name="common_pile_peps", description=description
    )
