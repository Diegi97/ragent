import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

from ragent_core.retrievers.base import BaseRetriever
from ragent_core.retrievers.chunking import DOCUMENT_ID_KEY
from ragent_core.retrievers.document import Document, RetrievalResult
from ragent_core.retrievers.mode import RetrievalMode
from ragent_core.retrievers.retriever import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_RERANKER_MODEL_NAME,
    TurbopufferRetriever,
)

logger = logging.getLogger(__name__)


_REGEX_META = re.compile(r"([\\.^$|?*+()\[\]{}])")
_UNSUPPORTED_REGEX = (
    (re.compile(r"\(\?(?:=|!|<=|<!)"), "lookaround"),
    (re.compile(r"\\[1-9]"), "backreferences"),
    (re.compile(r"\\g<|\(\?P="), "backreferences"),
    (re.compile(r"\(\?\("), "conditional groups"),
    (re.compile(r"\(\?>"), "atomic groups"),
)


def _escape_server_regex(value: str) -> str:
    """Escape only regex metacharacters using Rust-regex-compatible escapes."""
    return _REGEX_META.sub(r"\\\1", value)


def _validate_server_regex(pattern: str) -> None:
    for detector, feature in _UNSUPPORTED_REGEX:
        if detector.search(pattern):
            raise ValueError(
                f"Turbopuffer regex scans do not support {feature}: {pattern!r}"
            )


class AgentRetriever:
    """Document repository + agent-friendly tools.

    The :class:`AgentRetriever` is layered over a prebuilt
    :class:`TurbopufferRetriever`. Chunk records carry a source-document
    ``document_id`` column, and a companion ``<table>_documents`` table holds
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
        retriever: BaseRetriever,
        retrieval_mode: Optional[RetrievalMode] = None,
    ) -> None:
        self._retriever = retriever
        self._retrieval_mode = (
            RetrievalMode(retrieval_mode) if retrieval_mode is not None else None
        )
        self._search_lock: Optional[threading.Lock] = None

    @classmethod
    def from_turbopuffer_index(
        cls,
        namespace: str = "default",
        model_name: Optional[str] = DEFAULT_EMBEDDING_MODEL_NAME,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        reranker_model_name: Optional[str] = DEFAULT_RERANKER_MODEL_NAME,
        rerank_threshold: float = 0.0,
        top_rerank: int = 50,
        rerank_batch_size: int = 8,
        embedding_service_url: Optional[str] = None,
        reranker_service_url: Optional[str] = None,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID_RERANKED,
    ) -> "AgentRetriever":
        """Load an agent retriever from a Turbopuffer logical namespace.

        BM25 mode skips both embedding and reranker backend loading; dense and
        hybrid modes load only the backends their retrieval pipelines need.
        """
        retrieval_mode = RetrievalMode(retrieval_mode)
        needs_embeddings = retrieval_mode is not RetrievalMode.BM25
        needs_reranker = retrieval_mode is RetrievalMode.HYBRID_RERANKED
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
        )
        return cls(base_retriever, retrieval_mode=retrieval_mode)

    @property
    def retriever(self) -> BaseRetriever:
        return self._retriever

    @property
    def retrieval_mode(self) -> Optional[RetrievalMode]:
        return self._retrieval_mode

    def get_document(self, doc_id: Any, table_name: str) -> Optional[Document]:
        """Return the full :class:`Document` whose id is ``doc_id``.

        This is a point lookup in the corpus's companion document namespace.
        """
        try:
            return self._retriever.get_document(doc_id, table_name=table_name)
        except Exception:
            logger.exception(
                "Failed to query document namespace for document id=%s", doc_id
            )
            return None

    def retrieve(
        self,
        query: str,
        table_name: str,
        top_k: int = 50,
        **kwargs,
    ) -> List[RetrievalResult]:
        if self._retrieval_mode is not None:
            kwargs["retrieval_mode"] = self._retrieval_mode
        results = self._retriever.retrieve(
            query, table_name=table_name, top_k=top_k, **kwargs
        )
        for result in results:
            metadata = result.metadata if result.metadata is not None else {}
            metadata.setdefault(DOCUMENT_ID_KEY, result.id)
            result.metadata = metadata
        return results

    def _get_search_lock(self) -> threading.Lock:
        if self._search_lock is None:
            self._search_lock = threading.Lock()
        return self._search_lock

    @staticmethod
    def _document_id_and_title(res: RetrievalResult) -> Tuple[Any, str]:
        """Resolve the source document id/title for a chunk result."""
        metadata = res.metadata or {}
        return metadata.get(DOCUMENT_ID_KEY, res.id), (res.title or "")

    def search_tool(self, queries: List[str], table_name: str) -> str:
        """Search the indexed corpus for the most relevant documents.

        Returns an XML-formatted string with search results grouped by query.
        Snippets are taken from the matched chunks; ids and titles refer to
        the source document.
        """
        logger.debug("Search tool called with queries: %s", queries)

        with self._get_search_lock():
            xml_parts = ["<search_results>"]
            for query in queries:
                retrieval_results = self.retrieve(
                    query, top_k=10, table_name=table_name
                )

                xml_parts.append(f'<query value="{escape(query)}">')
                for res in retrieval_results:
                    doc_id, title = self._document_id_and_title(res)

                    xml_parts.append("<result>")
                    xml_parts.append(f"<id>{escape(str(doc_id))}</id>")
                    xml_parts.append(f"<title>{escape(title)}</title>")
                    xml_parts.append(f"<snippet>{escape(res.content)}</snippet>")
                    xml_parts.append("</result>")
                xml_parts.append("</query>")
            xml_parts.append("</search_results>")
            return "\n".join(xml_parts)

    def read_tool(self, doc_ids: List[Any], table_name: str) -> str:
        """Retrieve full document content by document ID.

        Args:
            doc_ids: List of document IDs. Limited to the first 3 to avoid
                token rate limits.
        """
        logger.debug("Read tool called with doc_ids: %s", doc_ids)

        xml_parts = ["<documents>"]
        for doc_id in doc_ids[:3]:
            doc = self.get_document(doc_id, table_name=table_name)
            if doc is not None:
                content = escape(doc.content or "")
            else:
                content = escape(
                    f"Error: Document with id '{doc_id}' not found in corpus."
                )

            xml_parts.append(f"<document id={doc_id}>")
            xml_parts.append(content)
            xml_parts.append("</document>")

        xml_parts.append("</documents>")
        return "\n".join(xml_parts)

    @staticmethod
    def _build_scan_regex(
        pattern: str,
        fixed_string: bool,
        case_sensitive: bool,
    ) -> str:
        """Build a Turbopuffer-compatible regex used for server prefiltering."""
        server_pattern = _escape_server_regex(pattern) if fixed_string else pattern
        if not fixed_string:
            _validate_server_regex(pattern)
        return server_pattern if case_sensitive else f"(?i){server_pattern}"

    def _scan_chunk_rows(
        self,
        pattern: str,
        fixed_string: bool,
        case_sensitive: bool,
        table_name: str,
    ) -> List[Tuple[str, Any, str]]:
        """Fetch ``(chunk_text, document_id, document_title)`` rows to scan.

        Scans the chunk ``content`` column rather than full documents, and only
        projects the small top-level ``document_id`` column. The trade-off is
        that a match straddling two chunks is not found.
        """
        server_regex = self._build_scan_regex(pattern, fixed_string, case_sensitive)
        return self._retriever.scan_chunks(table_name, server_regex)

    def text_scan_tool(
        self,
        pattern: str,
        table_name: str,
        fixed_string: bool = True,
        case_sensitive: bool = False,
        max_results: int = 25,
        snippet_chars: int = 200,
    ) -> str:
        """Scan the indexed chunk text for a regex or fixed-string match.

        Matches are aggregated back to their source documents and returned as
        an XML-formatted string, ranked by match count. Because the scan runs
        over chunk text, a match that straddles a chunk boundary is not found.
        """
        logger.debug(
            "Text scan tool called with pattern: %s, fixed_string: %s, case_sensitive: %s",
            pattern,
            fixed_string,
            case_sensitive,
        )
        if not pattern:
            return ""

        if fixed_string:
            if case_sensitive:
                needle = pattern

                def match_count(text: str) -> int:
                    return text.count(needle) if text else 0

                def find_first(text: str) -> int:
                    return text.find(needle) if text else -1

            else:
                needle = pattern.lower()

                def match_count(text: str) -> int:
                    return text.lower().count(needle) if text else 0

                def find_first(text: str) -> int:
                    return text.lower().find(needle) if text else -1

        else:
            _validate_server_regex(pattern)
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags=flags)
            except re.error as exc:
                raise ValueError(f"Invalid regular expression: {exc}") from exc

            def match_count(text: str) -> int:
                return sum(1 for _ in regex.finditer(text)) if text else 0

            def find_first(text: str) -> int:
                if not text:
                    return -1
                match = regex.search(text)
                return match.start() if match else -1

        rows = self._scan_chunk_rows(
            pattern, fixed_string, case_sensitive, table_name=table_name
        )

        # Aggregate matching chunks back to their source document: the total
        # match count drives ranking, and the chunk with the most matches
        # supplies the snippet.
        aggregated: Dict[Any, Dict[str, Any]] = {}
        for content, document_id, title in rows:
            count = match_count(content)
            if count <= 0:
                continue
            first_idx = find_first(content)
            key = document_id if document_id is not None else title
            entry = aggregated.get(key)
            if entry is None:
                aggregated[key] = {
                    "id": key,
                    "title": title,
                    "total": count,
                    "best_count": count,
                    "snippet_text": content,
                    "snippet_idx": first_idx,
                }
            else:
                entry["total"] += count
                if count > entry["best_count"]:
                    entry["best_count"] = count
                    entry["snippet_text"] = content
                    entry["snippet_idx"] = first_idx

        if not aggregated:
            return ""

        # Highest total match count first; document id as a stable tiebreak.
        ranked = sorted(aggregated.values(), key=lambda e: (-e["total"], str(e["id"])))
        top = ranked[:max_results]

        xml_parts = []
        for entry in top:
            text = entry["snippet_text"]
            first_idx = entry["snippet_idx"]
            if first_idx < 0:
                snippet = ""
            else:
                half = max(10, snippet_chars // 2)
                start = max(0, first_idx - half)
                end = min(len(text), first_idx + half)
                snippet = text[start:end]

            xml_parts.append("<match>")
            xml_parts.append(f"<id>{escape(str(entry['id']))}</id>")
            xml_parts.append(f"<title>{escape(entry['title'] or '')}</title>")
            xml_parts.append(f"<snippet>{escape(snippet)}</snippet>")
            xml_parts.append("</match>")

        result_xml = "\n".join(xml_parts)

        logger.debug(
            "Text scan matched %d documents (fixed_string=%s, case_sensitive=%s)",
            len(aggregated),
            fixed_string,
            case_sensitive,
        )
        return result_xml
