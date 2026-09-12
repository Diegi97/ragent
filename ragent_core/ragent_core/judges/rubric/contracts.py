from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from pydantic import BaseModel, Field

from ragent_core.judges.criteria import Verdict


class JudgeCriterion(BaseModel):
    id: str
    text: str
    weight: float = Field(default=1.0, gt=0)


class CriterionVerdict(BaseModel):
    id: str
    reason: str
    verdict: Verdict

    @classmethod
    def parse_xml(cls, text: str) -> list[CriterionVerdict]:
        """Extract the last valid ``<criteria>`` block from a model response.

        Prose and Markdown fences outside the XML are ignored. Nested markup inside a
        field is reduced to its combined text, so the reason is free to span lines.
        """

        parsed: list[CriterionVerdict] | None = None
        for match in _CRITERIA_XML_RE.finditer(text):
            block = match.group()
            verdicts: list[CriterionVerdict] = []
            try:
                root = ET.fromstring(block)
                for element in root.findall("./criterion"):
                    values: dict[str, str] = {}
                    for field in ("id", "reason", "verdict"):
                        child = element.find(field)
                        if child is not None:
                            values[field] = "".join(child.itertext()).strip()
                    verdicts.append(cls.model_validate(values))
            except (ET.ParseError, ValueError):
                try:
                    verdicts = cls._parse_unescaped_xml(block)
                except ValueError:
                    continue

            if verdicts:
                parsed = verdicts

        if parsed is None:
            raise ValueError(f"judge returned no parseable criteria XML: {text!r}")
        return parsed

    @classmethod
    def _parse_unescaped_xml(cls, block: str) -> list[CriterionVerdict]:
        """Parse judge fields as text when XML entities were not escaped."""

        verdicts: list[CriterionVerdict] = []
        for criterion_match in _CRITERION_XML_RE.finditer(block):
            body = criterion_match.group(1)
            values: dict[str, str] = {}
            for field, pattern in _FIELD_XML_RE.items():
                if match := pattern.search(body):
                    values[field] = match.group(1).strip()
            verdicts.append(cls.model_validate(values))
        return verdicts


_CRITERIA_XML_RE = re.compile(
    r"<criteria(?:\s[^>]*)?>.*?</criteria\s*>",
    re.DOTALL,
)


_CRITERION_XML_RE = re.compile(
    r"<criterion(?:\s[^>]*)?>(.*?)</criterion\s*>",
    re.DOTALL,
)


_FIELD_XML_RE = {
    field: re.compile(
        rf"<{field}(?:\s[^>]*)?>(.*?)</{field}\s*>",
        re.DOTALL,
    )
    for field in ("id", "reason", "verdict")
}
