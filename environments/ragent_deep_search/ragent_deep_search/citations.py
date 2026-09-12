import re
from dataclasses import dataclass

_CITATION_RE = re.compile(
    r"\[(?:doc|docs)\s+(?P<ids>\d+(?:\s*,\s*\d+)*)\]",
    re.IGNORECASE,
)


_SOURCES_HEADING_RE = re.compile(r"^## Sources\s*$", re.MULTILINE)


def citation_ids(text: str) -> list[int]:
    return [
        int(doc_id)
        for match in _CITATION_RE.finditer(text)
        for doc_id in match.group("ids").split(",")
    ]


@dataclass(frozen=True)
class GroundedCitations:
    document_ids: tuple[int, ...]

    @classmethod
    def from_response(
        cls, response: str, evidence_ids: set[int]
    ) -> "GroundedCitations | None":
        headings = list(_SOURCES_HEADING_RE.finditer(response))
        if not headings:
            return None
        heading = headings[-1]
        inline_ids = citation_ids(response[: heading.start()])
        source_ids = citation_ids(response[heading.end() :])
        expected = tuple(dict.fromkeys(inline_ids))
        if (
            not inline_ids
            or tuple(source_ids) != expected
            or not set(inline_ids) <= evidence_ids
        ):
            return None
        return cls(expected)
