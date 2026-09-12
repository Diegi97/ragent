import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from data_pipelines.pipelines.deep_search_task_generation.facts import (
    FactRequestMetadata,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.fact_extraction_output.diagnostics import (
    ParseDiagnostics,
)


def parse_batch_output_files(
    file_paths: Sequence[Path],
) -> tuple[dict[str, str], ParseDiagnostics]:
    responses: dict[str, str] = {}
    conflicts: set[str] = set()
    diagnostics = ParseDiagnostics()
    for path in file_paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Batch output record must be an object")
                except (ValueError, json.JSONDecodeError) as exc:
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
                custom_id = record.get("custom_id")
                if not isinstance(custom_id, str) or not custom_id.strip():
                    diagnostics.missing_custom_ids += 1
                    diagnostics.failures.append(
                        {
                            "stage": "batch_output_parsing",
                            "file": str(path),
                            "line": line_number,
                            "error": "Response has no valid custom_id",
                        }
                    )
                    continue
                custom_id = custom_id.strip()
                response = record.get("response")
                choices = (
                    response.get("choices") if isinstance(response, Mapping) else None
                )
                if (
                    not isinstance(choices, list)
                    or not choices
                    or not isinstance(choices[0], Mapping)
                ):
                    diagnostics.missing_choices += 1
                    diagnostics.failures.append(
                        {
                            "stage": "batch_output_parsing",
                            "custom_id": custom_id,
                            "error": "Response contains no choices",
                        }
                    )
                    continue
                message = choices[0].get("message")
                content = (
                    message.get("content") if isinstance(message, Mapping) else None
                )
                if (
                    not isinstance(content, str)
                    or not content.strip()
                    or message.get("refusal")
                ):
                    diagnostics.invalid_contents += 1
                    diagnostics.failures.append(
                        {
                            "stage": "batch_output_parsing",
                            "custom_id": custom_id,
                            "error": "Response contains no usable text",
                        }
                    )
                    continue
                if custom_id in conflicts:
                    continue
                if custom_id in responses:
                    diagnostics.duplicate_custom_ids += 1
                    if responses[custom_id] == content:
                        continue
                    diagnostics.conflicting_custom_ids += 1
                    conflicts.add(custom_id)
                    responses.pop(custom_id)
                    diagnostics.failures.append(
                        {
                            "stage": "batch_output_parsing",
                            "custom_id": custom_id,
                            "error": "Conflicting duplicate responses",
                        }
                    )
                    continue
                responses[custom_id] = content
    return responses, diagnostics


def load_batch_input_metadata(path: Path) -> dict[str, FactRequestMetadata]:
    metadata: dict[str, FactRequestMetadata] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = FactRequestMetadata.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed prepare input at line {line_number}: {exc}"
                ) from exc
            if record.custom_id in metadata:
                raise ValueError(
                    f"Duplicate prepare input custom_id {record.custom_id!r} at line {line_number}"
                )
            metadata[record.custom_id] = record
    return metadata
