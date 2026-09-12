import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from data_pipelines.artifacts.append import append_jsonl
from data_pipelines.artifacts.io import write_json, write_jsonl


def test_failed_replacement_preserves_previous_artifact(tmp_path):
    path = tmp_path / "metadata.json"
    write_json(path, {"complete": True})
    with pytest.raises(ValueError):
        write_json(path, {"score": float("nan")})
    assert json.loads(path.read_text()) == {"complete": True}
    assert sorted(file.name for file in tmp_path.iterdir()) == ["metadata.json"]


def test_concurrent_appends_do_not_interleave_records(tmp_path):
    path = tmp_path / "records.jsonl"
    lock = tmp_path / ".records.lock"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda i: append_jsonl(path, {"id": i, "text": "x" * 2000}, lock),
                range(20),
            )
        )
    assert sorted(
        json.loads(line)["id"] for line in path.read_text().splitlines()
    ) == list(range(20))


def test_jsonl_generator_failure_preserves_old_artifact(tmp_path):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"old": True}])

    def broken_records():
        yield {"new": True}
        raise RuntimeError("broken source")

    with pytest.raises(RuntimeError):
        write_jsonl(path, broken_records())
    assert json.loads(path.read_text()) == {"old": True}
