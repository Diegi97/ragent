import re
from dataclasses import dataclass
from typing import Any, Dict
from xml.sax.saxutils import escape

_REGEX_META = re.compile(r"([\\.^$|?*+()\[\]{}])")
_UNSUPPORTED_REGEX = (
    (re.compile(r"\(\?(?:=|!|<=|<!)"), "lookaround"),
    (re.compile(r"\\[1-9]"), "backreferences"),
    (re.compile(r"\\g<|\(\?P="), "backreferences"),
    (re.compile(r"\(\?\("), "conditional groups"),
    (re.compile(r"\(\?>"), "atomic groups"),
)


@dataclass(frozen=True)
class TextScan:
    pattern: str
    fixed_string: bool
    case_sensitive: bool
    server_regex: str
    regex: re.Pattern[str] | None

    @classmethod
    def prepare(
        cls, pattern: str, fixed_string: bool, case_sensitive: bool
    ) -> "TextScan":
        regex = None
        if fixed_string:
            server_pattern = _REGEX_META.sub(r"\\\1", pattern)
        else:
            for detector, feature in _UNSUPPORTED_REGEX:
                if detector.search(pattern):
                    raise ValueError(
                        f"Turbopuffer regex scans do not support {feature}: {pattern!r}"
                    )
            try:
                regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"Invalid regular expression: {exc}") from exc
            server_pattern = pattern
        return cls(
            pattern,
            fixed_string,
            case_sensitive,
            server_pattern if case_sensitive else f"(?i){server_pattern}",
            regex,
        )

    def render(
        self, rows: list[tuple[str, Any, str]], max_results: int, snippet_chars: int
    ) -> str:
        pattern, fixed_string, case_sensitive = (
            self.pattern,
            self.fixed_string,
            self.case_sensitive,
        )

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
            regex = self.regex
            assert regex is not None

            def match_count(text: str) -> int:
                return sum(1 for _ in regex.finditer(text)) if text else 0

            def find_first(text: str) -> int:
                if not text:
                    return -1
                match = regex.search(text)
                return match.start() if match else -1

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
            document_match = aggregated.get(key)
            if document_match is None:
                aggregated[key] = {
                    "id": key,
                    "title": title,
                    "total": count,
                    "best_count": count,
                    "snippet_text": content,
                    "snippet_idx": first_idx,
                }
            else:
                document_match["total"] += count
                if count > document_match["best_count"]:
                    document_match["best_count"] = count
                    document_match["snippet_text"] = content
                    document_match["snippet_idx"] = first_idx

        if not aggregated:
            return ""

        # Highest total match count first; document id as a stable tiebreak.
        ranked = sorted(
            aggregated.values(), key=lambda match: (-match["total"], str(match["id"]))
        )
        top_document_matches = ranked[:max_results]

        xml_parts = []
        for document_match in top_document_matches:
            text = document_match["snippet_text"]
            first_idx = document_match["snippet_idx"]
            if first_idx < 0:
                snippet = ""
            else:
                half = max(10, snippet_chars // 2)
                start = max(0, first_idx - half)
                end = min(len(text), first_idx + half)
                snippet = text[start:end]

            xml_parts.append("<match>")
            xml_parts.append(f"<id>{escape(str(document_match['id']))}</id>")
            xml_parts.append(f"<title>{escape(document_match['title'] or '')}</title>")
            xml_parts.append(f"<snippet>{escape(snippet)}</snippet>")
            xml_parts.append("</match>")

        result_xml = "\n".join(xml_parts)

        return result_xml
