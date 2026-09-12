"""Shared outcomes of deep-search QA and rubric generation."""

from enum import StrEnum


class GenerationStatus(StrEnum):
    FAILED = "failed"
    EMPTY = "empty"
    COMPLETED = "completed"
    PARTIAL = "partial"

    @classmethod
    def for_output(
        cls, generated: int, requested: int, source_degraded: bool = False
    ) -> "GenerationStatus":
        if source_degraded:
            return cls.PARTIAL if generated else cls.FAILED
        if generated == requested:
            return cls.COMPLETED
        return cls.PARTIAL if generated else cls.EMPTY
