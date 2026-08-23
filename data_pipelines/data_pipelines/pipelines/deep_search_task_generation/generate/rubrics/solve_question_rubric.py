"""Run one Verifiers solver rollout and report criterion-level judgments."""

import argparse
import asyncio
import copy
import json
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    QuestionRubricRecord,
    question_rubric_sha256,
    validate_question_rubric_file,
)

EVALUATION_CONFIG_ENV = "RAGENT_EVALUATION_CONFIG"
DATA_SOURCE_ENV = "RAGENT_DATA_SOURCE"
DEEP_SEARCH_SOURCE_ENV = "RAGENT_DEEP_SEARCH_SOURCE"
AUDITS_DIRECTORY_ENV = "RAGENT_AUDITS_DIRECTORY"
SOLVER_MODEL_ENV = "RAGENT_SOLVER_MODEL"
CITATION_RE = re.compile(r"\[(?:doc|docs)\s+(\d+(?:\s*,\s*\d+)*)\]", re.I)


@dataclass(frozen=True)
class SolverSettings:
    evaluation_config: Path
    data_source: str
    deep_search_source: Path
    audits_directory: Path
    solver_model: str


@dataclass(frozen=True)
class SolverRuntime:
    settings: SolverSettings
    evaluation_template: dict[str, Any]
    verifiers: Any
    eval_config_type: type[Any]
    run_eval: Any


def _validate_settings(values: dict[str, str | None]) -> SolverSettings:
    missing = [name for name, value in values.items() if not (value or "").strip()]
    if missing:
        raise ValueError(
            "required environment variables are not set: " + ", ".join(missing)
        )

    evaluation_config = Path(values[EVALUATION_CONFIG_ENV] or "").expanduser().resolve()
    deep_search_source = (
        Path(values[DEEP_SEARCH_SOURCE_ENV] or "").expanduser().resolve()
    )
    audits_directory = Path(values[AUDITS_DIRECTORY_ENV] or "").expanduser().resolve()
    if not evaluation_config.is_file():
        raise FileNotFoundError(
            f"evaluation config does not exist: {evaluation_config}"
        )
    if not deep_search_source.is_dir():
        raise FileNotFoundError(
            f"deep-search source directory does not exist: {deep_search_source}"
        )
    if not audits_directory.is_dir():
        raise FileNotFoundError(f"audits directory does not exist: {audits_directory}")
    return SolverSettings(
        evaluation_config=evaluation_config,
        data_source=(values[DATA_SOURCE_ENV] or "").strip(),
        deep_search_source=deep_search_source,
        audits_directory=audits_directory,
        solver_model=(values[SOLVER_MODEL_ENV] or "").strip(),
    )


def _load_runtime() -> SolverRuntime:
    settings = _validate_settings(
        {
            variable: os.getenv(variable)
            for variable in (
                EVALUATION_CONFIG_ENV,
                DATA_SOURCE_ENV,
                DEEP_SEARCH_SOURCE_ENV,
                AUDITS_DIRECTORY_ENV,
                SOLVER_MODEL_ENV,
            )
        }
    )
    load_dotenv(settings.evaluation_config.with_name(".env"), override=True)
    if str(settings.deep_search_source) not in sys.path:
        sys.path.insert(0, str(settings.deep_search_source))

    import verifiers.v1 as vf
    from verifiers.v1.cli.eval.runner import run_eval
    from verifiers.v1.configs.cli.eval import EvalConfig

    evaluation_template = tomllib.loads(
        settings.evaluation_config.read_text(encoding="utf-8")
    )
    return SolverRuntime(
        settings=settings,
        evaluation_template=evaluation_template,
        verifiers=vf,
        eval_config_type=EvalConfig,
        run_eval=run_eval,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _audit_path(candidate: Path, runtime: SolverRuntime) -> Path:
    return runtime.settings.audits_directory / f"{candidate.name}.solver.json"


def _candidate_dataset(
    candidate: Path,
    record: QuestionRubricRecord,
    digest: str,
    runtime: SolverRuntime,
) -> tuple[Path, Path]:
    run_directory = runtime.settings.audits_directory / candidate.name / digest
    run_directory.mkdir(parents=True, exist_ok=True)
    dataset_path = run_directory / "candidate.jsonl"
    payload = record.model_dump(mode="json")
    payload["data_source"] = runtime.settings.data_source
    dataset_path.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_json(
        run_directory / "metadata.json",
        {"prepare_config": {"data_source": runtime.settings.data_source}},
    )
    return dataset_path, run_directory / "evaluation"


def _eval_config(
    dataset_path: Path,
    output_directory: Path,
    runtime: SolverRuntime,
) -> Any:
    raw = copy.deepcopy(runtime.evaluation_template)
    raw.update(
        {
            "model": runtime.settings.solver_model,
            "num_tasks": 1,
            "num_rollouts": 1,
            "max_concurrent": 1,
            "shuffle": False,
            "push": False,
            "rich": False,
            "server": False,
            "output_dir": str(output_directory),
        }
    )
    taskset = raw.setdefault("env", {}).setdefault("taskset", {})
    taskset.update(
        {
            "dataset_path": str(dataset_path),
            "split": "test",
            "num_tasks": 1,
        }
    )
    return runtime.eval_config_type.model_validate(raw)


def _judge_verdicts(trace: Any) -> dict[str, dict[str, str]]:
    verdicts: dict[str, dict[str, str]] = {}
    for response in trace.info.get("judge", []):
        for verdict in response.get("parsed") or []:
            criterion_id = str(verdict.get("id") or "")
            if criterion_id:
                verdicts[criterion_id] = {
                    "reason": str(verdict.get("reason") or ""),
                    "verdict": str(verdict.get("verdict") or ""),
                }
    return verdicts


def _citation_ids(answer: str) -> list[int]:
    return list(
        dict.fromkeys(
            int(doc_id)
            for match in CITATION_RE.finditer(answer)
            for doc_id in match.group(1).split(",")
        )
    )


def _report(record: QuestionRubricRecord, trace: Any, digest: str) -> dict[str, Any]:
    if not trace.ok:
        detail = trace.last_error.message if trace.last_error is not None else "unknown"
        raise RuntimeError(f"solver rollout failed: {detail}")
    verdicts = _judge_verdicts(trace)
    judgments: list[dict[str, Any]] = []
    passed = 0
    for index, criterion in enumerate(record.rubric, start=1):
        criterion_id = f"C-{index:03d}"
        metric = trace.metrics.get(f"rubric/{criterion_id}")
        if metric is None:
            raise RuntimeError(f"solver trace is missing metric rubric/{criterion_id}")
        passed_criterion = metric == 1.0
        passed += int(passed_criterion)
        judgment = verdicts.get(criterion_id, {})
        judgments.append(
            {
                "id": criterion_id,
                "criterion": criterion.criterion,
                "doc_ids": criterion.doc_ids,
                "passed": passed_criterion,
                "verdict": judgment.get("verdict", ""),
                "reason": judgment.get("reason", ""),
            }
        )
    total = len(judgments)
    answer = trace.last_reply
    return {
        "ok": True,
        "candidate_sha256": digest,
        "question": record.question,
        "answer": answer,
        "cited_doc_ids": _citation_ids(answer),
        "judgments": judgments,
        "criteria_passed": passed,
        "criteria_total": total,
        "percent_passed": round(100 * passed / total, 2),
    }


async def _solve(candidate: Path) -> dict[str, Any]:
    runtime = _load_runtime()
    candidate = candidate.expanduser().resolve()
    digest = question_rubric_sha256(candidate)
    audit_path = _audit_path(candidate, runtime)
    _write_json(audit_path, {"ok": False, "candidate_sha256": digest})
    record = validate_question_rubric_file(candidate)
    dataset_path, output_directory = _candidate_dataset(
        candidate, record, digest, runtime
    )
    config = _eval_config(dataset_path, output_directory, runtime)
    previous_directory = Path.cwd()
    try:
        os.chdir(runtime.settings.evaluation_config.parent)
        environment = runtime.verifiers.load_environment(config.env)
        episodes = await runtime.run_eval(environment, config)
    finally:
        os.chdir(previous_directory)
    if len(episodes) != 1 or len(episodes[0].traces) != 1:
        raise RuntimeError("solver evaluation did not return exactly one rollout")
    if not episodes[0].ok:
        detail = episodes[0].last_error
        raise RuntimeError(f"solver evaluation failed: {detail}")
    result = _report(record, episodes[0].traces[0], digest)
    _write_json(audit_path, result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = _parse_args()
        print(json.dumps(asyncio.run(_solve(args.candidate)), ensure_ascii=False))
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
