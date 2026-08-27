import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from filelock import FileLock

from data_pipelines.pipelines.deep_search_task_generation.config import (
    FACT_RESPONSES_RELATIVE_PATH,
    DeepSearchTaskGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.models import PreparePaths
from data_pipelines.pipelines.deep_search_task_generation.project import utc_timestamp
from ragent_core.retrievers.document import RetrievalResult
from ragent_core.types import Concept


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (slug or "source")[:80]


def initialize_prepare_output(
    config: DeepSearchTaskGenerationConfig,
    run_id: str,
) -> PreparePaths:
    directory = config.output_root / (
        f"{_slug(config.data_source)}_{utc_timestamp()}_"
        f"{config.num_entities}e_{run_id[:8]}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    retrieval_debug_directory = directory / "retrieval_debug"
    retrieval_debug_directory.mkdir()
    paths = PreparePaths(
        directory=directory,
        fact_responses=directory / FACT_RESPONSES_RELATIVE_PATH,
        retrieval_debug_directory=retrieval_debug_directory,
        entities=directory / "entities.jsonl",
        fact_requests=directory / "fact_requests.jsonl",
        failures=directory / "failures.jsonl",
        metadata=directory / "prepare_metadata.json",
        lock=directory / ".records.lock",
    )
    for path in (paths.entities, paths.fact_requests, paths.failures):
        path.touch(exist_ok=False)
    return paths


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


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temporary, path)


def write_retrieval_debug(
    path: Path,
    entity_name: str,
    top_k: int,
    chunks: Sequence[RetrievalResult],
) -> None:
    """Write the ungrouped, reranker-filtered chunks for one entity."""
    write_json(
        path,
        {
            "entity": entity_name,
            "top_k": top_k,
            "final_chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.metadata.get("document_id", chunk.id),
                    "title": chunk.title,
                    "cross_encoder_score": chunk.score,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
                for chunk in sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
            ],
        },
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fp:
        json.dump(value, fp, indent=2, ensure_ascii=False, default=str)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        value = json.load(fp)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def count_jsonl(path: Path) -> int:
    with path.open("rb") as fp:
        return sum(1 for line in fp if line.strip())


def entity_to_dict(entity: Concept) -> dict[str, Any]:
    return {
        "name": entity.name,
        "data_source": entity.data_source,
        "doc_id": entity.doc_id,
        "importance": entity.importance,
        "info": entity.info,
    }
