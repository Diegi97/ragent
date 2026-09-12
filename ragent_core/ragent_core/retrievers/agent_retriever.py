import logging
import threading
from typing import Any, List, Optional

from ragent_core.retrievers.base import AgentRetrieverBackend
from ragent_core.retrievers.document import DOCUMENT_ID_KEY, Document, RetrievalResult
from ragent_core.retrievers.mode import RetrievalMode
from ragent_core.retrievers.retriever import TurbopufferRetriever
from ragent_core.retrievers.settings import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_LOGICAL_NAMESPACE,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_THRESHOLD,
    DEFAULT_RERANKER_MODEL_NAME,
    DEFAULT_TOP_K,
    TOP_RERANK,
)
from ragent_core.retrievers.text_scan import TextScan
from ragent_core.retrievers.tool_protocol import (
    DEFAULT_SCAN_RESULTS,
    DEFAULT_SCAN_SNIPPET_CHARS,
    MAX_READ_DOCUMENTS,
    MAX_SEARCH_QUERIES,
    SEARCH_TOP_K,
    document_xml,
    search_query_xml,
)
from ragent_core.retrievers.transport import create_turbopuffer_client

logger = logging.getLogger(__name__)


class AgentRetriever:
    """Document repository + agent-friendly tools.

    The :class:`AgentRetriever` is layered over a prebuilt
    :class:`TurbopufferRetriever`. Chunk records carry a source-document
    ``document_id`` column, and the catalog-resolved document namespace holds
    the full documents, so documents are recovered with a single indexed lookup
    instead of an in-memory index or a full scan of the chunks table.

    - ``retrieve()`` returns chunk-level :class:`RetrievalResult` records.
      ``result.metadata[DOCUMENT_ID_KEY]`` points back at the source document;
      ``result.id`` is the chunk's own id.
    - ``read_tool`` resolves full documents by id from the documents table.
      ``text_scan_tool`` scans the indexed chunk text (a match that straddles a
      chunk boundary is not found) and reports the documents the matches belong
      to.
    """

    def __init__(
        self,
        retriever: AgentRetrieverBackend,
        retrieval_mode: Optional[RetrievalMode] = None,
    ) -> None:
        self._retriever = retriever
        self._retrieval_mode = (
            RetrievalMode(retrieval_mode) if retrieval_mode is not None else None
        )
        self._search_lock = threading.Lock()

    @classmethod
    def from_turbopuffer_index(
        cls,
        namespace: str = DEFAULT_LOGICAL_NAMESPACE,
        model_name: Optional[str] = DEFAULT_EMBEDDING_MODEL_NAME,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        reranker_model_name: Optional[str] = DEFAULT_RERANKER_MODEL_NAME,
        rerank_threshold: float = DEFAULT_RERANK_THRESHOLD,
        top_rerank: int = TOP_RERANK,
        rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        embedding_service_url: Optional[str] = None,
        reranker_service_url: Optional[str] = None,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID_RERANKED,
        turbopuffer_api_key: Optional[str] = None,
    ) -> "AgentRetriever":
        """Load an agent retriever from a Turbopuffer logical namespace.

        BM25 mode skips both embedding and reranker backend loading; dense and
        hybrid modes load only the backends their retrieval pipelines need.
        """
        retrieval_mode = RetrievalMode(retrieval_mode)
        needs_embeddings = retrieval_mode.needs_embeddings
        needs_reranker = retrieval_mode.needs_reranker
        if (
            needs_reranker
            and reranker_model_name is None
            and reranker_service_url is None
        ):
            raise ValueError(
                "HYBRID_RERANKED retrieval requires a reranker model or service."
            )

        base_retriever = TurbopufferRetriever.load_index(
            namespace=namespace,
            model_name=model_name or DEFAULT_EMBEDDING_MODEL_NAME,
            device=device,
            trust_remote_code=trust_remote_code,
            reranker_model_name=reranker_model_name if needs_reranker else None,
            rerank_threshold=rerank_threshold,
            top_rerank=top_rerank,
            rerank_batch_size=rerank_batch_size,
            embedding_service_url=(embedding_service_url if needs_embeddings else None),
            reranker_service_url=reranker_service_url if needs_reranker else None,
            load_embedding_backend=needs_embeddings,
            client=(
                create_turbopuffer_client(api_key=turbopuffer_api_key)
                if turbopuffer_api_key is not None
                else None
            ),
        )
        return cls(base_retriever, retrieval_mode=retrieval_mode)

    @property
    def retriever(self) -> AgentRetrieverBackend:
        return self._retriever

    @property
    def retrieval_mode(self) -> Optional[RetrievalMode]:
        return self._retrieval_mode

    def get_document(self, doc_id: Any, table_name: str) -> Optional[Document]:
        """Return the full :class:`Document` whose id is ``doc_id``.

        This is a point lookup in the corpus's companion document namespace.
        """
        return self._retriever.get_document(doc_id, table_name=table_name)

    def retrieve(
        self,
        query: str,
        table_name: str,
        top_k: int = DEFAULT_TOP_K,
        **kwargs,
    ) -> List[RetrievalResult]:
        if self._retrieval_mode is not None:
            kwargs["retrieval_mode"] = self._retrieval_mode
        results = self._retriever.retrieve(
            query, table_name=table_name, top_k=top_k, **kwargs
        )
        for result in results:
            metadata = result.metadata if result.metadata is not None else {}
            result.metadata = metadata
            metadata.setdefault(DOCUMENT_ID_KEY, result.source_document_id)
        return results

    def search_tool(self, queries: List[str], table_name: str) -> str:
        """Return bounded search evidence grouped by query."""
        with self._search_lock:
            parts = ["<search_results>"]
            for query in queries[:MAX_SEARCH_QUERIES]:
                results = self.retrieve(
                    query, table_name=table_name, top_k=SEARCH_TOP_K
                )
                parts.append(search_query_xml(query, results))
            return "\n".join([*parts, "</search_results>"])

    def read_tool(self, doc_ids: List[Any], table_name: str) -> str:
        """Return bounded full-document evidence; backend failures propagate."""
        parts = [
            document_xml(doc_id, self.get_document(doc_id, table_name))
            for doc_id in doc_ids[:MAX_READ_DOCUMENTS]
        ]
        return "\n".join(["<documents>", *parts, "</documents>"])

    def text_scan_tool(
        self,
        pattern: str,
        table_name: str,
        fixed_string: bool = True,
        case_sensitive: bool = False,
        max_results: int = DEFAULT_SCAN_RESULTS,
        snippet_chars: int = DEFAULT_SCAN_SNIPPET_CHARS,
    ) -> str:
        """Scan chunks and aggregate matching evidence by source document."""
        if max_results < 1 or snippet_chars < 1:
            raise ValueError("max_results and snippet_chars must be positive")
        if not pattern:
            return ""
        scan = TextScan.prepare(pattern, fixed_string, case_sensitive)
        rows = self._retriever.scan_chunks(table_name, scan.server_regex)
        return scan.render(rows, max_results, snippet_chars)
