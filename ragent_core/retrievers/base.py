import logging
import re
import threading
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)


class BaseRetriever:
    """Common utilities for retriever tool interfaces."""

    _documents: Dict[Any, dict]
    _search_lock: Optional[threading.Lock] = None

    @property
    def documents(self) -> Dict[Any, dict]:
        """Return corpus documents keyed by document ID."""
        return getattr(self, "_documents", {})

    def _get_search_lock(self) -> threading.Lock:
        if self._search_lock is None:
            self._search_lock = threading.Lock()
        return self._search_lock

    def search_tool(self, queries: List[str]) -> str:
        """
        Search the indexed corpus for the most relevant documents.

        Returns an XML-formatted string with search results grouped by query.
        """
        logger.debug("Search tool called with queries: %s", queries)

        with self._get_search_lock():
            xml_parts = ["<search_results>"]
            for query in queries:
                retrieval_results = self.retrieve(query, top_k=10)

                xml_parts.append(f"<query value=\"{escape(query)}\">")
                for res in retrieval_results:
                    doc = self.documents.get(res.doc_id, {})
                    doc_id = doc.get("doc_id", res.doc_id)
                    title = doc.get("title", "") or ""

                    xml_parts.append("<result>")
                    xml_parts.append(f"<id>{escape(str(doc_id))}</id>")
                    xml_parts.append(f"<title>{escape(title)}</title>")
                    xml_parts.append(f"<snippet>{escape(res.text)}</snippet>")
                    xml_parts.append("</result>")
                xml_parts.append("</query>")
            xml_parts.append("</search_results>")
            return "\n".join(xml_parts)

    def read_tool(self, doc_ids: List[int]) -> str:
        """
        Retrieve document text by ID and return an XML-formatted string.

        Args:
            doc_ids: List of document IDs. If more than 3 IDs are provided, only the first 3 will be processed.
        """
        logger.debug("Read tool called with doc_ids: %s", doc_ids)

        xml_parts = ["<documents>"]
        # Limit to maximum 3 documents to avoid token rate limits
        for doc_id in doc_ids[:3]:
            doc = self.documents.get(doc_id, {})
            if doc is not None:
                text = doc.get("text", "") or ""
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
        for doc in self.documents.values():
            text = doc.get("text", "") or ""
            count = match_count(text)
            if count <= 0:
                continue
            first_idx = find_first(text)
            scored_results.append((count, -first_idx, doc.get("id")))

        if not scored_results:
            return ""

        scored_results.sort(reverse=True)
        top = scored_results[:max_results]

        xml_parts = []
        for _, neg_first_idx, doc_id in top:
            doc = self.documents[doc_id]
            text = doc.get("text", "")
            first_idx = -neg_first_idx
            if first_idx < 0:
                snippet = ""
            else:
                half = max(10, snippet_chars // 2)
                start = max(0, first_idx - half)
                end = min(len(text), first_idx + half)
                snippet = text[start:end]

            title = doc.get("title", "") or ""

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
