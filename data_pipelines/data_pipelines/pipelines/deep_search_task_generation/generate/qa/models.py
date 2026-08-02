from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FinalizePaths:
    directory: Path
    raw_directory: Path
    entity_facts: Path
    qas: Path
    failures: Path
    metadata: Path
    lock: Path


@dataclass(frozen=True)
class EntityQASummary:
    entity_name: str
    requested: int
    generated: int
    crashed: bool = False
    error: str | None = None
    phoenix_trace_id: str = ""
