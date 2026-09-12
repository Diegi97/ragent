import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        value = json.load(fp)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def count_jsonl(path: Path) -> int:
    with path.open("rb") as fp:
        return sum(1 for line in fp if line.strip())
