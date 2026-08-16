from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Union

KNOWN_FIELDS = {"id", "title", "content", "vector", "metadata", "document_id"}


@dataclass
class Document:
    """Single corpus document used by every retriever."""

    id: Union[int, str]
    title: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)
    document_id: Optional[Union[int, str]] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Document":
        if "id" not in data:
            raise ValueError("Document dict must include an 'id' field")

        doc_id = data["id"]
        if doc_id is None or (isinstance(doc_id, str) and not doc_id.strip()):
            raise ValueError("Document id must be a non-empty value")

        title = data.get("title") or ""
        content = data.get("content")
        if content is None:
            content = data.get("text", "") or ""

        metadata = dict(data.get("metadata") or {})
        for key, value in data.items():
            if key in KNOWN_FIELDS or key == "text":
                continue
            metadata.setdefault(key, value)

        document_id = data.get("document_id")

        return cls(
            id=doc_id,
            title=title,
            content=content,
            metadata=metadata,
            document_id=document_id,
        )

    def to_dict(self) -> dict:
        """Serialize to a plain ``dict`` compatible with :meth:`from_dict`."""
        data = {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
        }
        if self.document_id is not None:
            data["document_id"] = self.document_id
        return data

    @classmethod
    def from_hf_dataset(cls, dataset: Any) -> List["Document"]:
        """Convert a HuggingFace ``Dataset`` (or any iterable of dict rows)
        into a list of ``Document``.

        The ``text`` field is accepted as an alias for ``content``.
        """
        return [cls.from_dict(row) for row in dataset]


DocumentLike = Union[Document, Mapping[str, Any]]


@dataclass
class RetrievalResult:
    """A retrieved document with its score.

    Has the exact same fields as :class:`Document` plus a ``score``.
    """

    id: Union[int, str]
    score: float = 0.0
    title: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain ``dict``."""
        return {
            "id": self.id,
            "score": self.score,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
        }


def normalize_documents(
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
