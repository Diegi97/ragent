from typing import Any, List, Optional, Protocol

from ragent_core.retrievers.document import Document, DocumentId, RetrievalResult
from ragent_core.retrievers.mode import RetrievalMode
from ragent_core.retrievers.settings import DEFAULT_TOP_K


class BaseRetriever:
    """Search-only interface over a prebuilt corpus. Storage is backend-owned."""

    def retrieve(
        self,
        query: str,
        table_name: str,
        top_k: int = DEFAULT_TOP_K,
        retrieval_mode: Optional[RetrievalMode] = None,
        **kwargs,
    ) -> List[RetrievalResult]:
        raise NotImplementedError


class AgentRetrieverBackend(Protocol):
    """Capabilities required by the agent's search, read, and scan tools."""

    def retrieve(
        self,
        query: str,
        table_name: str,
        top_k: int = DEFAULT_TOP_K,
        retrieval_mode: RetrievalMode | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]: ...

    def get_document(self, doc_id: DocumentId, table_name: str) -> Document | None: ...

    def scan_chunks(
        self, table_name: str, server_regex: str
    ) -> list[tuple[str, DocumentId | None, str]]: ...
