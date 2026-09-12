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
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from data_pipelines.artifacts.io import write_json
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.audit_contract import (
    AUDITS_DIRECTORY_ENV,
    DATA_SOURCE_ENV,
    EVALUATION_CONFIG_ENV,
    AuditPaths,
    RetrievalAudit,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation.audits import (
    begin_candidate_audit,
)
from ragent_core.retrievers.tool_protocol import (
    MAX_READ_DOCUMENTS,
    MAX_SEARCH_QUERIES,
    ToolName,
    search_document_ids,
)


def _runtime_types() -> tuple[type[Any], type[Any]]:
    from ragent_deep_search.toolset import RagentToolset
    from ragent_deep_search.toolset.config import RagentToolsetConfig

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


def _audit_path(candidate: Path) -> Path:
    return AuditPaths(_audits_directory(), candidate.name).retrieval


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


async def _call_tool(command: str, command_values: list[str]) -> str:
    toolset_type, _ = _runtime_types()
    config = await asyncio.to_thread(_load_tool_config)
    toolset = toolset_type(config)
    await toolset.setup()
    if command == ToolName.SEARCH:
        return await toolset.search(command_values, table_name=_data_source())
    if command == ToolName.READ:
        return await toolset.read(
            [int(value) for value in command_values],
            table_name=_data_source(),
        )
    raise ValueError(f"unsupported corpus command: {command}")


async def _run_probe(candidate: Path) -> dict[str, Any]:
    candidate = candidate.expanduser().resolve()
    audit_path = await asyncio.to_thread(_audit_path, candidate)
    record, digest = await asyncio.to_thread(
        begin_candidate_audit, candidate, audit_path
    )
    search_output = await _call_tool(ToolName.SEARCH, [record.question])
    retrieved_doc_ids = search_document_ids(search_output)
    supporting_doc_ids = list(record.doc_ids)
    missing_doc_ids = [
        doc_id for doc_id in supporting_doc_ids if doc_id not in retrieved_doc_ids
    ]
    probe_passed = bool(missing_doc_ids)
    result = RetrievalAudit.model_validate(
        {
            "ok": True,
            "candidate_sha256": digest,
            "question": record.question,
            "supporting_doc_ids": supporting_doc_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
            "missing_doc_ids": missing_doc_ids,
            "all_supporting_docs_retrieved": not missing_doc_ids,
            "too_easy": not probe_passed,
            "probe_passed": probe_passed,
        }
    ).model_dump(mode="json")
    await asyncio.to_thread(write_json, audit_path, result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("candidate", type=Path)
    search = subparsers.add_parser(ToolName.SEARCH)
    search.add_argument("queries", nargs="+")
    read = subparsers.add_parser(ToolName.READ)
    read.add_argument("doc_ids", nargs="+")
    return parser.parse_args()


async def _main() -> dict[str, Any] | str:
    args = _parse_args()
    if args.command == "probe":
        return await _run_probe(args.candidate)
    command_values = args.queries if args.command == ToolName.SEARCH else args.doc_ids
    limit = (
        MAX_SEARCH_QUERIES if args.command == ToolName.SEARCH else MAX_READ_DOCUMENTS
    )
    if len(command_values) > limit:
        raise ValueError(f"{args.command} accepts at most {limit} values")
    return await _call_tool(args.command, command_values)


def main() -> None:
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


if __name__ == "__main__":
    main()
