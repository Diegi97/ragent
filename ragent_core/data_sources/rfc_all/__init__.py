import logging
from pathlib import Path

from datasets import Dataset, DatasetInfo

from ragent_core.data_sources import (
    DataSourceSpec,
    filter_by_word_count,
    keep_only_core_columns,
)

logger = logging.getLogger(__name__)


def _title_from_path(path: Path) -> str:
    stem = path.stem
    if stem.lower().startswith("rfc") and stem[3:].isdigit():
        return f"RFC {stem[3:]}"
    return stem


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def load_data_source() -> DataSourceSpec:
    logger.info(
        "Loading and normalizing RFC documents from data/RFC-all for QA generation"
    )

    base_dir = Path("data") / "RFC-all"
    if not base_dir.exists():
        raise FileNotFoundError(
            "RFC directory not found at data/RFC-all. Ensure the RFC text files are present."
        )

    files = sorted(p for p in base_dir.rglob("*.txt") if p.is_file())
    if not files:
        raise ValueError("No RFC .txt files found under data/RFC-all")

    records = []
    for idx, path in enumerate(files):
        text = _read_text(path)
        title = _title_from_path(path)
        if title and text:
            merged_text = f"# {title}\n\n{text}"
        else:
            merged_text = text or title
        records.append(
            {
                "id": idx,
                "title": title,
                "text": merged_text,
            }
        )

    dataset = Dataset.from_list(records, info=DatasetInfo(dataset_name="rfc_all"))
    dataset = filter_by_word_count(dataset)
    dataset = keep_only_core_columns(dataset)
    description = (
        "This dataset contains IETF Requests for Comments (RFCs) as plain text files. "
        "Documents are formal technical standards, protocols, and informational memos "
        "written in long-form prose with structured sections."
    )
    return DataSourceSpec(dataset=dataset, name="rfc_all", description=description)
