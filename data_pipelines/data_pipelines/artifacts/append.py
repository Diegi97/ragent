import json
import os
from pathlib import Path
from typing import Any

from filelock import FileLock


def append_jsonl(path: Path, value: dict[str, Any], lock_path: Path) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, default=str) + "\n").encode()
    with FileLock(lock_path):
        with path.open("ab", buffering=0) as fp:
            remaining = memoryview(encoded)
            while remaining:
                written = fp.write(remaining)
                if written is None or written <= 0:
                    raise OSError(f"Failed to append a record to {path}.")
                remaining = remaining[written:]
            fp.flush()
            os.fsync(fp.fileno())
