import json
import logging
from pathlib import Path

from ragent_core.judges.evaluation.models import GroundTruthExample

logger = logging.getLogger(__name__)


def resolve_traces_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "traces.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Teacher-labeled traces file not found: {path}")
    return path


def load_ground_truth(path: Path) -> list[GroundTruthExample]:
    examples: list[GroundTruthExample] = []
    seen_example_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc.msg}"
                ) from exc
            if not isinstance(trace, dict):
                raise ValueError(f"Line {line_number} of {path} must be a JSON object")
            try:
                example = GroundTruthExample.from_trace(trace)
            except ValueError as exc:
                raise ValueError(f"Invalid trace on line {line_number}: {exc}") from exc
            if example.example_id in seen_example_ids:
                raise ValueError(
                    f"Duplicate trace ID {example.example_id!r} on line {line_number}"
                )
            seen_example_ids.add(example.example_id)
            examples.append(example)

    if not examples:
        raise ValueError(f"Teacher-labeled traces file is empty: {path}")
    return examples
