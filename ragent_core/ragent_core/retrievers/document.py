from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, TypedDict, Union

DocumentId = int | str
DOCUMENT_ID_KEY = "document_id"

KNOWN_FIELDS = {"id", "title", "content", "vector", "metadata", DOCUMENT_ID_KEY}


@dataclass
class Document:
    """Single corpus document used by every retriever."""

    id: DocumentId
    title: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)
    document_id: Optional[DocumentId] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Document":
        if "id" not in data:
            raise ValueError("Document dict must include an 'id' field")

        doc_id = cls.validate_id(data["id"])

        title = data.get("title")
        if title is None:
            title = ""
        content = data.get("content")
        if content is None:
            content = data.get("text")
            if content is None:
                content = ""

        if not isinstance(title, str) or not isinstance(content, str):
            raise TypeError("Document title and content must be strings")
        raw_metadata = data.get("metadata")
        if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
            raise TypeError("Document metadata must be an object")
        metadata = dict(raw_metadata or {})
        for key, value in data.items():
            if key in KNOWN_FIELDS or key == "text":
                continue
            metadata.setdefault(key, value)

        document_id = data.get(DOCUMENT_ID_KEY)
        if document_id is not None:
            document_id = cls.validate_id(document_id)

        return cls(
            id=doc_id,
            title=title,
            content=content,
            metadata=metadata,
            document_id=document_id,
        )

    @staticmethod
    def validate_id(value: Any) -> DocumentId:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, str))
            or isinstance(value, str)
            and not value.strip()
        ):
            raise ValueError("Document id must be a non-empty integer or string")
        return value

    def to_dict(self) -> dict:
        """Serialize to a plain ``dict`` compatible with :meth:`from_dict`."""
        data = {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
        }
        if self.document_id is not None:
            data[DOCUMENT_ID_KEY] = self.document_id
        return data

    @classmethod
    def from_hf_dataset(cls, dataset: Any) -> List["Document"]:
        """Convert a HuggingFace ``Dataset`` (or any iterable of dict rows)
        into a list of ``Document``.

        The ``text`` field is accepted as an alias for ``content``.
        """
        return [cls.from_dict(row) for row in dataset]

    @classmethod
    def normalize_many(
        cls,
        documents: Iterable[DocumentLike],
    ) -> List[Document]:
        """Accept an iterable of ``Document`` or dict rows and return a list of
        ``Document`` with unique ids."""
        normalized: List[Document] = []
        seen_ids: set = set()
        for raw in documents:
            if isinstance(raw, Document):
                doc = raw
            elif isinstance(raw, Mapping):
                doc = Document.from_dict(raw)
            else:
                raise TypeError(
                    f"Documents must be Document or mapping; got {type(raw).__name__}"
                )

            if doc.id in seen_ids:
                raise ValueError(f"Duplicate document id detected: {doc.id!r}")
            seen_ids.add(doc.id)
            normalized.append(doc)

        return normalized


DocumentLike = Union[Document, Mapping[str, Any]]


class RetrievalResultData(TypedDict):
    id: DocumentId
    score: float
    title: str
    content: str
    metadata: dict[str, Any]


@dataclass
class RetrievalResult:
    """A retrieved document with its score.

    Has the exact same fields as :class:`Document` plus a ``score``.
    """

    id: DocumentId
    score: float = 0.0
    title: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def parent_document_id(self) -> DocumentId | None:
        """Chunk parent identity, absent for an unchunked result."""
        return self.metadata.get(DOCUMENT_ID_KEY)

    @property
    def source_document_id(self) -> DocumentId:
        """Full-document identity; unchunked backends use the result ID itself."""
        return self.metadata.get(DOCUMENT_ID_KEY, self.id)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RetrievalResult":
        document = Document.from_dict(
            {
                key: value[key]
                for key in ("id", "title", "content", "metadata")
                if key in value
            }
        )
        if DOCUMENT_ID_KEY in document.metadata:
            Document.validate_id(document.metadata[DOCUMENT_ID_KEY])
        score = value.get("score", 0.0)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError("Retrieval result score must be finite and numeric")
        return cls(
            id=document.id,
            title=document.title,
            content=document.content,
            metadata=document.metadata,
            score=float(score),
        )

    def to_dict(self) -> RetrievalResultData:
        """Serialize to a plain ``dict``."""
        return {
            "id": self.id,
            "score": self.score,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
        }
