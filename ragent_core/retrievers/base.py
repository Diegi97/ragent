import logging
import re
import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)


class BaseRetriever:
    """Common utilities for retriever tool interfaces."""

    id_field = "id"
    title_field = "title"
    text_field = "text"

    _doc_index_by_id: Dict[Any, int]
    _search_lock: Optional[threading.Lock] = None

    def _iter_corpus(self) -> Iterable[Tuple[int, Mapping[str, Any]]]:
        """Yield (index, document) pairs for the retriever corpus."""
        raise NotImplementedError

    def _get_doc_by_index(self, index: int) -> Mapping[str, Any]:
        """Return a document by integer index."""
        raise NotImplementedError

    def _get_doc_id(self, doc: Mapping[str, Any], index: int) -> Any:
        return doc.get(self.id_field, index)

    def _get_doc_title(self, doc: Mapping[str, Any]) -> str:
        return doc.get(self.title_field, "") or ""

    def _get_doc_text(self, doc: Mapping[str, Any]) -> str:
        return doc.get(self.text_field, "") or ""

    def _ensure_doc_index_by_id(self) -> None:
        if getattr(self, "_doc_index_by_id", None) is None:
            self._doc_index_by_id = {}
        if self._doc_index_by_id:
            return

        for index, doc in self._iter_corpus():
            doc_id = self._get_doc_id(doc, index)
            if doc_id not in self._doc_index_by_id:
                self._doc_index_by_id[doc_id] = index

    def _get_search_lock(self) -> threading.Lock:
        if self._search_lock is None:
            self._search_lock = threading.Lock()
        return self._search_lock

    def _default_snippet(self, text: str, max_words: int = 50) -> str:
        words = text.split()[:max_words]
        return " ".join(words) + "..." if words else ""

    def search_tool(self, queries: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """
        Search the indexed corpus for the most relevant documents.
        """
        logger.debug("Search tool called with queries: %s", queries)
        self._ensure_doc_index_by_id()
        with self._get_search_lock():
            all_results: Dict[str, List[Dict[str, str]]] = {}
            for query in queries:
                retrieval_results = self.retrieve(query, top_k=10)

                formatted_results = []
                for res in retrieval_results:
                    row_index = self._doc_index_by_id.get(res.doc_id)
                    if row_index is None and isinstance(res.doc_id, int):
                        row_index = self._doc_index_by_id.get(str(res.doc_id))
                    if row_index is None:
                        continue

                    doc = self._get_doc_by_index(row_index)
                    text = self._get_doc_text(doc)
                    snippet = res.text or self._default_snippet(text)

                    formatted_results.append(
                        {
                            "id": res.doc_id,
                            "title": self._get_doc_title(doc),
                            "snippet": snippet,
                        }
                    )
                all_results[query] = formatted_results
            return all_results

    def read_tool(self, doc_ids: List[int]) -> str:
        """
        Retrieve document text by ID and return an XML-formatted string.

        Args:
            doc_ids: List of document IDs. If more than 3 IDs are provided, only the first 3 will be processed.
        """
        logger.debug("Read tool called with doc_ids: %s", doc_ids)
        self._ensure_doc_index_by_id()

        xml_parts = ["<documents>"]
        # Limit to maximum 3 documents to avoid token rate limits
        for doc_id in doc_ids[:3]:
            row_index = self._doc_index_by_id.get(doc_id)

            if row_index is None and isinstance(doc_id, int):
                row_index = self._doc_index_by_id.get(str(doc_id))

            if row_index is not None:
                doc = self._get_doc_by_index(row_index)
                text = self._get_doc_text(doc)
                content = escape(text or "")
            else:
                content = escape(
                    f"Error: Document with id '{doc_id}' not found in corpus."
                )

            xml_parts.append(f"<document id={doc_id}>")
            xml_parts.append(content)
            xml_parts.append("</document>")

        xml_parts.append("</documents>")
        return "\n".join(xml_parts)

    def text_scan_tool(
        self,
        pattern: str,
        fixed_string: bool = True,
        case_sensitive: bool = False,
        max_results: int = 25,
        snippet_chars: int = 200,
    ) -> str:
        """
        Scan all documents for a regex or fixed-string substring match.

        Returns top results (scored by match count) as XML formatted string:
        <match><id>...</id><title>...</title><snippet>...</snippet></match>...
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

            regex = None
        else:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags=flags)

            def match_count(text: str) -> int:
                return sum(1 for _ in regex.finditer(text)) if text else 0

            def find_first(text: str) -> int:
                if not text:
                    return -1
                match = regex.search(text)
                return match.start() if match else -1

        scored_results: List[tuple[int, int, int]] = []
        for row_index, doc in self._iter_corpus():
            text = self._get_doc_text(doc)
            count = match_count(text)
            if count <= 0:
                continue
            first_idx = find_first(text)
            scored_results.append((count, -first_idx, row_index))

        if not scored_results:
            return ""

        scored_results.sort(reverse=True)
        top = scored_results[:max_results]

        xml_parts = []
        for _, neg_first_idx, row_index in top:
            doc = self._get_doc_by_index(row_index)
            text = self._get_doc_text(doc)
            first_idx = -neg_first_idx
            if first_idx < 0:
                snippet = ""
            else:
                half = max(10, snippet_chars // 2)
                start = max(0, first_idx - half)
                end = min(len(text), first_idx + half)
                snippet = text[start:end]

            doc_id = self._get_doc_id(doc, row_index)
            title = self._get_doc_title(doc)

            xml_parts.append("<match>")
            xml_parts.append(f"<id>{escape(str(doc_id))}</id>")
            xml_parts.append(f"<title>{escape(title)}</title>")
            xml_parts.append(f"<snippet>{escape(snippet)}</snippet>")
            xml_parts.append("</match>")

        result_xml = "\n".join(xml_parts)

        logger.debug(
            "Text scan returned %d/%d matches (fixed_string=%s, case_sensitive=%s)",
            len(top),
            len(scored_results),
            fixed_string,
            case_sensitive,
        )
        # Returning the results as an XML formatted string is needed for the LLM APIs
        return result_xml
