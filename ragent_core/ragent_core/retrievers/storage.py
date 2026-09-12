import json
import math
from collections.abc import Mapping
from typing import Any

from ragent_core.retrievers.document import DOCUMENT_ID_KEY, Document, RetrievalResult

RELEVANCE_SCORE_FIELD = "_relevance_score"
SOURCE_INDEX_FIELD = "source_index"
METADATA_JSON_FIELD = "metadata_json"

REMOTE_FIELDS = (
    "title",
    "content",
    METADATA_JSON_FIELD,
    DOCUMENT_ID_KEY,
    SOURCE_INDEX_FIELD,
)


CHUNK_SCHEMA = {
    "title": {"type": "string", "filterable": False},
    "content": {
        "type": "string",
        "filterable": False,
        "regex": True,
        "full_text_search": {
            "tokenizer": "word_v4",
            "case_sensitive": False,
            "stemming": False,
            "remove_stopwords": False,
            "ascii_folding": False,
        },
    },
    METADATA_JSON_FIELD: {"type": "string", "filterable": False},
    DOCUMENT_ID_KEY: {"type": "uint", "filterable": False},
    SOURCE_INDEX_FIELD: {"type": "uint", "filterable": True},
}


DOCUMENT_SCHEMA = {
    "title": {"type": "string", "filterable": False},
    "content": {"type": "string", "filterable": False},
    METADATA_JSON_FIELD: {"type": "string", "filterable": False},
    DOCUMENT_ID_KEY: {"type": "uint", "filterable": False},
    SOURCE_INDEX_FIELD: {"type": "uint", "filterable": True},
}


def document_row(document: Document, source_index: int) -> dict[str, Any]:
    return {
        "id": document.id,
        "title": document.title,
        "content": document.content,
        METADATA_JSON_FIELD: json.dumps(
            document.metadata, ensure_ascii=False, allow_nan=False
        ),
        DOCUMENT_ID_KEY: None,
        SOURCE_INDEX_FIELD: source_index,
    }


def chunk_row(document: Document, source_index: int) -> dict[str, Any]:
    if type(document.document_id) is not int or document.document_id < 0:
        raise ValueError("Turbopuffer chunk document_id must be a non-negative integer")
    return {
        **document_row(document, source_index),
        DOCUMENT_ID_KEY: document.document_id,
    }


def deserialize_document(row: Mapping[str, Any]) -> Document:
    metadata_raw = row.get(METADATA_JSON_FIELD)
    if not isinstance(metadata_raw, str):
        raise ValueError("Stored metadata_json must be a string")
    try:
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Stored metadata is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Stored metadata must decode to an object")
    return Document.from_dict(
        {
            "id": row["id"],
            "title": row.get("title"),
            "content": row.get("content"),
            "metadata": metadata,
            DOCUMENT_ID_KEY: row.get(DOCUMENT_ID_KEY),
        }
    )


def deserialize_result(row: Mapping[str, Any]) -> RetrievalResult:
    document = deserialize_document(row)
    metadata = dict(document.metadata)
    parent_id = document.document_id
    if type(parent_id) is not int or parent_id < 0:
        raise ValueError("Stored chunk document_id must be a non-negative integer")
    if DOCUMENT_ID_KEY in metadata and (
        type(metadata[DOCUMENT_ID_KEY]) is not int
        or metadata[DOCUMENT_ID_KEY] != parent_id
    ):
        raise ValueError("Stored chunk metadata conflicts with its document_id")
    metadata[DOCUMENT_ID_KEY] = parent_id
    raw_score = row.get(RELEVANCE_SCORE_FIELD)
    if raw_score is None:
        raw_score = row.get("$dist", 0.0)
    if (
        isinstance(raw_score, bool)
        or not isinstance(raw_score, (int, float))
        or not math.isfinite(raw_score)
    ):
        raise ValueError("Stored retrieval score must be finite and numeric")
    return RetrievalResult(
        id=document.id,
        title=document.title,
        content=document.content,
        metadata=metadata,
        score=float(raw_score),
    )
