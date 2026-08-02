import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio
from prefect import task

from data_pipelines.pipelines.deep_search_task_generation.models import (
    EntityFactMemoryRecord,
    ParseDiagnostics,
)
from data_pipelines.pipelines.deep_search_task_generation.project import phoenix_project
from data_pipelines.pipelines.deep_search_task_generation.prompts import (
    parse_extracted_facts,
)
from data_pipelines.pipelines.deep_search_task_generation.services import (
    download_batch_dataset,
)
from data_pipelines.tracing import object_trace


def parse_batch_output_files(
    file_paths: Sequence[Path],
) -> tuple[dict[str, str], ParseDiagnostics]:
    responses: dict[str, str] = {}
    diagnostics = ParseDiagnostics()
    for path in file_paths:
        with path.open(encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    diagnostics.malformed_lines += 1
                    diagnostics.failures.append(
                        {
                            "stage": "batch_output_parsing",
                            "file": str(path),
                            "line": line_number,
                            "error": str(exc),
                        }
                    )
                    continue
                if not isinstance(record, dict):
                    diagnostics.malformed_lines += 1
                    diagnostics.failures.append(
                        {
                            "stage": "batch_output_parsing",
                            "file": str(path),
                            "line": line_number,
                            "error": "Batch output record is not a JSON object.",
                        }
                    )
                    continue
                custom_id = record.get("custom_id")
                if not custom_id:
                    diagnostics.missing_custom_ids += 1
                    continue
                custom_id = str(custom_id)
                response = record.get("response") or {}
                choices = (
                    response.get("choices") or []
                    if isinstance(response, Mapping)
                    else []
                )
                first_choice = choices[0] if choices else None
                if not isinstance(first_choice, Mapping):
                    diagnostics.missing_choices += 1
                    diagnostics.failures.append(
                        {
                            "stage": "batch_output_parsing",
                            "custom_id": custom_id,
                            "error": "Response contains no choices.",
                        }
                    )
                    continue
                if custom_id in responses:
                    diagnostics.duplicate_custom_ids += 1
                message = first_choice.get("message") or {}
                responses[custom_id] = (
                    str(message.get("content") or "")
                    if isinstance(message, Mapping)
                    else ""
                )
    return responses, diagnostics


def load_batch_input_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed prepare input at line {line_number}: {exc}"
                ) from exc
            custom_id = record.get("custom_id")
            if custom_id:
                metadata[str(custom_id)] = {
                    "entity_name": record.get("entity_name"),
                    "data_source": record.get("data_source"),
                    "doc_ids": record.get("doc_ids") or [],
                    "chunk_ids": record.get("chunk_ids") or [],
                }
    return metadata


def build_entity_facts_from_batch_output(
    responses: Mapping[str, str],
    metadata_by_key: Mapping[str, Mapping[str, Any]],
    diagnostics: ParseDiagnostics | None = None,
) -> list[EntityFactMemoryRecord]:
    diagnostics = diagnostics or ParseDiagnostics()
    records: dict[str, dict[str, Any]] = {}
    for custom_id, content in responses.items():
        metadata = metadata_by_key.get(custom_id)
        if metadata is None:
            diagnostics.unmatched_responses += 1
            diagnostics.failures.append(
                {
                    "stage": "batch_output_join",
                    "custom_id": custom_id,
                    "error": "No matching fact request metadata.",
                }
            )
            continue
        entity_name = str(metadata.get("entity_name") or "").strip()
        if not entity_name:
            diagnostics.failures.append(
                {
                    "stage": "batch_output_join",
                    "custom_id": custom_id,
                    "error": "Request metadata has no entity name.",
                }
            )
            continue
        current = records.setdefault(
            entity_name,
            {
                "data_source": str(metadata.get("data_source") or ""),
                "doc_ids": [],
                "facts": [],
                "statements": set(),
            },
        )
        for fact in parse_extracted_facts(content, entity_name=entity_name):
            key = " ".join(fact.statement.lower().split())
            if key and key not in current["statements"]:
                current["statements"].add(key)
                current["facts"].append(fact)
        for doc_id in metadata.get("doc_ids") or []:
            value = int(doc_id)
            if value not in current["doc_ids"]:
                current["doc_ids"].append(value)
    for custom_id in set(metadata_by_key).difference(responses):
        diagnostics.failures.append(
            {
                "stage": "batch_output_join",
                "custom_id": custom_id,
                "error": "No valid response was found for this fact request.",
            }
        )
    return [
        EntityFactMemoryRecord(
            entity_name=entity_name,
            data_source=value["data_source"],
            entity_doc_ids=tuple(value["doc_ids"]),
            facts=tuple(value["facts"]),
        )
        for entity_name, value in records.items()
    ]


@task(name="parse-fact-extraction-output", retries=0, persist_result=False)
async def parse_fact_output(
    file_paths: Sequence[Path], batch_input_path: Path
) -> tuple[dict[str, str], list[EntityFactMemoryRecord], ParseDiagnostics]:
    with object_trace(
        "parse-fact-extraction-output",
        {
            "file_paths": [str(path) for path in file_paths],
            "batch_input_path": str(batch_input_path),
        },
        {"batch.output_file_count": len(file_paths)},
        project_name=phoenix_project(),
    ) as root:
        responses, diagnostics = await anyio.to_thread.run_sync(
            parse_batch_output_files, file_paths
        )
        input_metadata = await anyio.to_thread.run_sync(
            load_batch_input_metadata, batch_input_path
        )
        entity_facts = await anyio.to_thread.run_sync(
            build_entity_facts_from_batch_output,
            responses,
            input_metadata,
            diagnostics,
        )
        root.set_output(
            {
                "response_count": len(responses),
                "entity_fact_count": len(entity_facts),
                "diagnostics": diagnostics.to_dict(),
            }
        )
        return responses, entity_facts, diagnostics


@task(name="download-fact-extraction-output", retries=0, persist_result=False)
async def download_output(
    dataset_name: str, output_directory: Path, timeout: float
) -> list[Path]:
    return await anyio.to_thread.run_sync(
        download_batch_dataset, dataset_name, output_directory, timeout
    )
