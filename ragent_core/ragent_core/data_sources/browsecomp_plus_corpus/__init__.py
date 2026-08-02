import logging
import re

from datasets import Dataset, load_dataset
from lingua import Language, LanguageDetectorBuilder

from ragent_core.data_sources import (
    DataSourceSpec,
    filter_by_word_count,
    keep_only_core_columns,
)

logger = logging.getLogger(__name__)

_FRONTMATTER_TITLE_RE = re.compile(r"^---\s*\ntitle:\s*(.+?)\s*\n", re.MULTILINE)

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH).build()
    return _detector


def _extract_title(text: str) -> str | None:
    m = _FRONTMATTER_TITLE_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def _transform(example: dict) -> dict:
    text = example.get("text") or ""
    return {
        "id": int(example["docid"]),
        "title": _extract_title(text),
        "text": text,
    }


def _is_english(example: dict) -> bool:
    text = example.get("text") or ""
    # Use a short prefix for speed; frontmatter + first paragraph is enough signal
    snippet = text[:500]
    return _get_detector().detect_language_of(snippet) == Language.ENGLISH


def load_data_source() -> DataSourceSpec:
    logger.info(
        "Loading Tevatron/browsecomp-plus-corpus and filtering for English texts"
    )

    raw = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    if not isinstance(raw, Dataset):
        raise TypeError(f"Expected Dataset, got {type(raw)}")

    dataset = raw.map(_transform, remove_columns=raw.column_names)
    dataset = dataset.filter(_is_english)
    dataset = filter_by_word_count(dataset)
    dataset = keep_only_core_columns(dataset)

    description = (
        "A curated corpus of English web documents used for deep research tasks."
    )
    return DataSourceSpec(
        dataset=dataset,
        name="Tevatron_browsecomp_plus_corpus",
        description=description,
    )
