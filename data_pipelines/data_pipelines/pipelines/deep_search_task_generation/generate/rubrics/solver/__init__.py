import asyncio
import copy
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from data_pipelines.artifacts.io import write_json
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.audit_contract import (
    AuditPaths,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.solver.config import (
    SOLVER_EPISODE_MAX_RETRIES,
    SOLVER_EPISODE_TIMEOUT_SECONDS,
    SolverSettings,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.solver.reports import (
    error_detail,
    solver_report,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation.audits import (
    begin_candidate_audit,
)
from ragent_core.artifacts.question_rubric import QuestionRubricRecord


@dataclass(frozen=True)
class SolverRuntime:
    settings: SolverSettings
    evaluation_template: dict[str, Any]
    verifiers: Any
    eval_config_type: type[Any]
    run_eval: Any

    @classmethod
    def from_environment(cls) -> "SolverRuntime":
        settings = SolverSettings.from_environment()
        load_dotenv(settings.evaluation_config.with_name(".env"), override=True)

        import verifiers.v1 as vf
        from verifiers.v1.cli.eval.runner import run_eval
        from verifiers.v1.configs.cli.eval import EvalConfig

        evaluation_template = tomllib.loads(
            settings.evaluation_config.read_text(encoding="utf-8")
        )
        return cls(
            settings=settings,
            evaluation_template=evaluation_template,
            verifiers=vf,
            eval_config_type=EvalConfig,
            run_eval=run_eval,
        )

    async def solve(self, candidate: Path) -> dict[str, Any]:
        candidate = candidate.expanduser().resolve()
        audit_path = self.audit_path(candidate)
        record, digest = await asyncio.to_thread(
            begin_candidate_audit, candidate, audit_path
        )
        dataset_path, output_directory = await asyncio.to_thread(
            self.candidate_dataset, candidate, record, digest
        )
        config = self.evaluation_config(dataset_path, output_directory)
        # Verifiers resolves relative environment paths against this standalone project.
        previous_directory = Path.cwd()
        try:
            os.chdir(self.settings.evaluation_config.parent)
            environment = await asyncio.to_thread(
                self.verifiers.load_environment, config.env
            )
            episodes = await self.run_eval(environment, config)
        finally:
            os.chdir(previous_directory)
        if len(episodes) != 1 or len(episodes[0].traces) != 1:
            raise RuntimeError("solver evaluation did not return exactly one rollout")
        if not episodes[0].ok:
            episode = episodes[0]
            detail = error_detail(episode)
            if detail == "unknown error":
                detail = error_detail(episode.traces[0])
            raise RuntimeError(f"solver evaluation failed: {detail}")
        result = solver_report(record, episodes[0].traces[0], digest)
        await asyncio.to_thread(write_json, audit_path, result)
        return result

    def audit_path(self, candidate: Path) -> Path:
        return AuditPaths(self.settings.audits_directory, candidate.name).solver

    def candidate_dataset(
        self, candidate: Path, record: QuestionRubricRecord, digest: str
    ) -> tuple[Path, Path]:
        run_directory = self.settings.audits_directory / candidate.name / digest
        run_directory.mkdir(parents=True, exist_ok=True)
        dataset_path = run_directory / "candidate.jsonl"
        payload = record.model_dump(mode="json")
        payload["data_source"] = self.settings.data_source
        dataset_path.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_json(
            run_directory / "metadata.json",
            {"prepare_config": {"data_source": self.settings.data_source}},
        )
        return dataset_path, run_directory / "evaluation"

    def evaluation_config(self, dataset_path: Path, output_directory: Path) -> Any:
        raw = copy.deepcopy(self.evaluation_template)
        raw.update(
            {
                "model": self.settings.solver_model,
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
        environment = raw.setdefault("env", {})
        environment.setdefault("timeout", {})["episode"] = (
            SOLVER_EPISODE_TIMEOUT_SECONDS
        )
        environment.setdefault("retries", {})["max_retries"] = (
            SOLVER_EPISODE_MAX_RETRIES
        )
        taskset = environment.setdefault("taskset", {})
        taskset.update(
            {
                "dataset_path": str(dataset_path),
                "split": "test",
                "num_tasks": 1,
            }
        )
        return self.eval_config_type.model_validate(raw)
