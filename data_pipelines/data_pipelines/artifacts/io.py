import json
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO


def write_json(path: Path, value: Any) -> None:
    with atomic_text_writer(path) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with atomic_text_writer(path) as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            stream.write("\n")


@contextmanager
def atomic_text_writer(path: Path) -> Iterator[TextIO]:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
