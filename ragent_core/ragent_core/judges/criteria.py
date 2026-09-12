from __future__ import annotations

from enum import StrEnum
from typing import Self


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        if isinstance(value, str):
            return cls.__members__.get(value.strip().upper())
        return None

    @classmethod
    def from_teacher(cls, value: str) -> "Verdict":
        aliases = {"yes": cls.PASS, "no": cls.FAIL}
        normalized = value.strip().casefold()
        return aliases[normalized] if normalized in aliases else cls(value)

    @property
    def score(self) -> float:
        return float(self is self.PASS)


def criterion_id(index: int) -> str:
    return f"C-{index:03d}"


def criterion_metric_name(identifier: str, reward_name: str = "rubric") -> str:
    return f"{reward_name}/{identifier}"
