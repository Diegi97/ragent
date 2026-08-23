"""Probe question retrieval and expose direct corpus search/read commands.

The probe command prints a structured JSON decision. Search and read print the
underlying XML tool output directly so a Pi agent can inspect it through Bash.
"""

import argparse
import asyncio
import json
import os
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    question_rubric_sha256,
    validate_question_rubric_file,
)

EVALUATION_CONFIG_ENV = "RAGENT_EVALUATION_CONFIG"
DATA_SOURCE_ENV = "RAGENT_DATA_SOURCE"
DEEP_SEARCH_SOURCE_ENV = "RAGENT_DEEP_SEARCH_SOURCE"
AUDITS_DIRECTORY_ENV = "RAGENT_AUDITS_DIRECTORY"


def _runtime_types() -> tuple[type[Any], type[Any]]:
    source = os.getenv(DEEP_SEARCH_SOURCE_ENV)
    if source and source not in sys.path:
        sys.path.insert(0, source)
    from ragent_deep_search.toolset import RagentToolset, RagentToolsetConfig

    return RagentToolset, RagentToolsetConfig


def _evaluation_config_path() -> Path:
    value = os.getenv(EVALUATION_CONFIG_ENV, "").strip()
    if not value:
        raise ValueError(f"{EVALUATION_CONFIG_ENV} is not set")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"evaluation config does not exist: {path}")
    return path


def _data_source() -> str:
    value = os.getenv(DATA_SOURCE_ENV, "").strip()
    if not value:
        raise ValueError(f"{DATA_SOURCE_ENV} is not set")
    return value


def _audits_directory() -> Path:
    value = os.getenv(AUDITS_DIRECTORY_ENV, "").strip()
    if not value:
        raise ValueError(f"{AUDITS_DIRECTORY_ENV} is not set")
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _audit_path(candidate: Path) -> Path:
    return _audits_directory() / f"{candidate.name}.retrieval.json"


def _load_tool_config() -> Any:
    config_path = _evaluation_config_path()
    load_dotenv(config_path.with_name(".env"), override=True)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    tools = raw.get("env", {}).get("taskset", {}).get("tools", {})
    _, config_type = _runtime_types()
    config = config_type.model_validate(tools)
    if config.env_file is not None and not config.env_file.is_absolute():
        config = config.model_copy(
            update={"env_file": (config_path.parent / config.env_file).resolve()}
        )
    return config


async def _call_tool(command: str, values: list[str]) -> str:
    toolset_type, _ = _runtime_types()
    toolset = toolset_type(_load_tool_config())
    await toolset.setup()
    if command == "search":
        return await toolset.search(values, table_name=_data_source())
    if command == "read":
        return await toolset.read(
            [int(value) for value in values],
            table_name=_data_source(),
        )
    raise ValueError(f"unsupported corpus command: {command}")


def _search_document_ids(search_output: str) -> list[int]:
    root = ET.fromstring(search_output)
    return list(
        dict.fromkeys(
            int(element.text.strip())
            for element in root.findall(".//result/id")
            if element.text and element.text.strip()
        )
    )


async def _run_probe(candidate: Path) -> dict[str, Any]:
    candidate = candidate.expanduser().resolve()
    digest = question_rubric_sha256(candidate)
    audit_path = _audit_path(candidate)
    failed_audit: dict[str, Any] = {
        "ok": False,
        "candidate_sha256": digest,
    }
    _write_json(audit_path, failed_audit)
    record = validate_question_rubric_file(candidate)
    search_output = await _call_tool("search", [record.question])
    retrieved_doc_ids = _search_document_ids(search_output)
    supporting_doc_ids = list(record.doc_ids)
    missing_doc_ids = [
        doc_id for doc_id in supporting_doc_ids if doc_id not in retrieved_doc_ids
    ]
    probe_passed = bool(missing_doc_ids)
    result = {
        "ok": True,
        "candidate_sha256": digest,
        "question": record.question,
        "supporting_doc_ids": supporting_doc_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
        "missing_doc_ids": missing_doc_ids,
        "all_supporting_docs_in_top_10": not missing_doc_ids,
        "too_easy": not probe_passed,
        "probe_passed": probe_passed,
    }
    _write_json(audit_path, result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("candidate", type=Path)
    search = subparsers.add_parser("search")
    search.add_argument("queries", nargs="+")
    read = subparsers.add_parser("read")
    read.add_argument("doc_ids", nargs="+")
    return parser.parse_args()


async def _main() -> dict[str, Any] | str:
    args = _parse_args()
    if args.command == "probe":
        return await _run_probe(args.candidate)
    values = args.queries if args.command == "search" else args.doc_ids
    if len(values) > 3:
        raise ValueError(f"{args.command} accepts at most three values")
    return await _call_tool(args.command, values)


if __name__ == "__main__":
    try:
        result = asyncio.run(_main())
        print(
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False)
        )
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
