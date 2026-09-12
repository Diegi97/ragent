from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParseDiagnostics:
    malformed_lines: int = 0
    missing_custom_ids: int = 0
    missing_choices: int = 0
    unmatched_responses: int = 0
    duplicate_custom_ids: int = 0
    conflicting_custom_ids: int = 0
    invalid_contents: int = 0
    missing_responses: int = 0
    unsupported_facts: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_integrity_errors(self) -> bool:
        return bool(self.failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "failures"
        }
