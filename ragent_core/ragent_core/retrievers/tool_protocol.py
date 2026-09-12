"""Bounded XML evidence returned by corpus tools."""

import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from enum import StrEnum
from xml.sax.saxutils import escape, quoteattr

from ragent_core.retrievers.document import (
    Document,
    DocumentId,
    RetrievalResult,
)


class ToolName(StrEnum):
    SEARCH = "search"
    READ = "read"
    TEXT_SCAN = "text_scan"


MAX_SEARCH_QUERIES = 3
SEARCH_TOP_K = 10
MAX_READ_DOCUMENTS = 3
DEFAULT_SCAN_RESULTS = 25
DEFAULT_SCAN_SNIPPET_CHARS = 200


_DOCUMENT_RE = re.compile(
    r"<document\s+id=(?:[\"'])?(?P<id>\d+)(?:[\"'])?\s*>"
    r"(?P<content>.*?)</document\s*>",
    re.DOTALL | re.IGNORECASE,
)


_SEARCH_RESULT_RE = re.compile(
    r"<result>.*?<id>\s*(?P<id>\d+)\s*</id>.*?</result\s*>",
    re.DOTALL | re.IGNORECASE,
)


_MISSING_DOCUMENT_PREFIX = "Error: Document with id"


def search_query_xml(query: str, results: Sequence[RetrievalResult]) -> str:
    parts = [f"<query value={quoteattr(query)}>"]
    for result in results:
        document_id = result.source_document_id
        parts.extend(
            [
                "<result>",
                f"<id>{escape(str(document_id))}</id>",
                f"<title>{escape(result.title)}</title>",
                f"<snippet>{escape(result.content)}</snippet>",
                "</result>",
            ]
        )
    return "\n".join([*parts, "</query>"])


def document_xml(document_id: DocumentId, document: Document | None) -> str:
    content = (
        document.content
        if document is not None
        else f"{_MISSING_DOCUMENT_PREFIX} '{document_id}' not found in corpus."
    )
    return (
        f"<document id={quoteattr(str(document_id))}>\n{escape(content)}\n</document>"
    )


def output_evidence_document_ids(tool_name: ToolName, output: str) -> set[int]:
    """Read current or historical tool evidence, excluding unsuccessful reads."""
    if tool_name == ToolName.SEARCH:
        return {int(match.group("id")) for match in _SEARCH_RESULT_RE.finditer(output)}
    if tool_name == ToolName.READ:
        return {
            int(match.group("id"))
            for match in _DOCUMENT_RE.finditer(output)
            if not match.group("content").lstrip().startswith(_MISSING_DOCUMENT_PREFIX)
        }
    return set()


def search_document_ids(search_output: str) -> list[int]:
    """Decode fresh, complete search XML for a numeric-document retrieval audit."""
    root = ET.fromstring(search_output)
    return list(
        dict.fromkeys(
            int(element.text.strip())
            for element in root.findall(".//result/id")
            if element.text and element.text.strip()
        )
    )
