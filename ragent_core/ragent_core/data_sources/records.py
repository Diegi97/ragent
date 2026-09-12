from typing import Any

DATA_SOURCES_DATASET_ID = "diegi97/ragent_data_sources"
CORE_COLUMNS = frozenset({"id", "title", "text"})


def source_record(
    document_id: int | str, title: str | None, text: str | None
) -> dict[str, Any]:
    title, text = title or "", text or ""
    if title and text and not text.lstrip().startswith(f"# {title}"):
        text = f"# {title}\n\n{text}"
    return {"id": document_id, "title": title, "text": text or title}
