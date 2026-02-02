import logging

from datasets import load_dataset

from ragent_core.data_sources import DataSourceSpec, filter_by_word_count

logger = logging.getLogger(__name__)


def _split_title_body(text: str) -> tuple[str, str]:
    lines = text.splitlines() if text else []
    if not lines:
        return "", ""
    title_index = None
    for idx, line in enumerate(lines):
        if line.strip():
            title_index = idx
            break
    if title_index is None:
        return "", ""
    title = lines[title_index].strip()
    body = "\n".join(lines[title_index + 1 :]).strip()
    return title, body


def load_data_source() -> DataSourceSpec:
    logger.info(
        "Loading and normalizing common-pile/uspto_filtered dataset for QA generation"
    )

    dataset = load_dataset("common-pile/uspto_filtered", split="train")

    def _formatter(example, index):
        text = example.get("text") or ""
        title, body = _split_title_body(text)
        if title and body:
            merged_text = f"# {title}\n\n{body}"
        else:
            merged_text = title or body
        return {
            "id": index,
            "title": title,
            "text": merged_text,
        }

    dataset = dataset.map(_formatter, with_indices=True)
    dataset = filter_by_word_count(dataset)
    description = (
        "This dataset contains U.S. patent documents with invention titles followed "
        "by formal patent text such as application details, claims, and descriptions."
    )
    return DataSourceSpec(
        dataset=dataset, name="common_pile_uspto_filtered", description=description
    )
