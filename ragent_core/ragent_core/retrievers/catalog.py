from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Self

from ragent_core.retrievers.settings import (
    DEFAULT_LOGICAL_NAMESPACE,
    DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX,
)

CATALOG_SCHEMA_VERSION = 1


def catalog_namespace(
    logical_namespace: str = DEFAULT_LOGICAL_NAMESPACE,
    namespace_prefix: str = DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX,
) -> str:
    return f"{namespace_prefix}.{logical_namespace}.catalog"


def corpus_namespaces(
    table_name: str,
    logical_namespace: str = DEFAULT_LOGICAL_NAMESPACE,
    namespace_prefix: str = DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX,
) -> tuple[str, str]:
    return (
        f"{namespace_prefix}.{logical_namespace}.{table_name}.chunks",
        f"{namespace_prefix}.{logical_namespace}.{table_name}.documents",
    )


@dataclass(frozen=True)
class CorpusCatalogEntry:
    table_name: str
    logical_namespace: str
    chunks_namespace: str
    documents_namespace: str
    schema_version: int
    chunk_count: int
    document_count: int
    vector_available: bool
    vector_dimensions: int
    embedding_model: str
    ready: bool

    def __post_init__(self) -> None:
        for name in (
            "table_name",
            "logical_namespace",
            "chunks_namespace",
            "documents_namespace",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Catalog {name} must be a non-empty string")
        for name in (
            "schema_version",
            "chunk_count",
            "document_count",
            "vector_dimensions",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Catalog {name} must be a non-negative integer")
        if type(self.ready) is not bool or type(self.vector_available) is not bool:
            raise ValueError(
                "Catalog readiness and vector availability must be booleans"
            )
        if not isinstance(self.embedding_model, str):
            raise ValueError("Catalog embedding_model must be a string")
        if self.vector_available and (
            self.vector_dimensions == 0 or not self.embedding_model.strip()
        ):
            raise ValueError(
                "Vector catalog entries require dimensions and an embedding model"
            )
        if not self.vector_available and (
            self.vector_dimensions != 0 or self.embedding_model
        ):
            raise ValueError("Lexical catalog entries cannot declare vector metadata")

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Self:
        try:
            return cls(**{name: row[name] for name in cls.__dataclass_fields__})
        except KeyError as exc:
            raise ValueError(f"Catalog entry is missing {exc.args[0]}") from exc

    def to_row(self) -> dict[str, Any]:
        return {"id": self.table_name, **asdict(self)}

    def require_identity(self, namespace: str, table_name: str) -> None:
        if self.logical_namespace != namespace or self.table_name != table_name:
            raise ValueError(
                f"Catalog identity does not match {namespace}/{table_name}"
            )


CATALOG_SCHEMA = {
    "table_name": {"type": "string", "filterable": True},
    "logical_namespace": {"type": "string", "filterable": True},
    "chunks_namespace": {"type": "string", "filterable": False},
    "documents_namespace": {"type": "string", "filterable": False},
    "schema_version": {"type": "uint", "filterable": True},
    "chunk_count": {"type": "uint", "filterable": False},
    "document_count": {"type": "uint", "filterable": False},
    "vector_available": {"type": "bool", "filterable": True},
    "vector_dimensions": {"type": "uint", "filterable": False},
    "embedding_model": {"type": "string", "filterable": False},
    "ready": {"type": "bool", "filterable": True},
}
