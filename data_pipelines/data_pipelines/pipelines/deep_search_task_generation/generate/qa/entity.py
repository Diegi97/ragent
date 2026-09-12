import asyncio
from pathlib import Path
from typing import Any

import anyio
from prefect import flow, task

from data_pipelines.artifacts.append import append_jsonl
from data_pipelines.pipelines.deep_search_task_generation.facts import (
    EntityFactMemoryRecord,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.candidates import (
    generate_qa_candidate,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.models import (
    ComplexityQuota,
    EntityQASummary,
    FinalizePaths,
)
from data_pipelines.pipelines.deep_search_task_generation.project import (
    phoenix_project,
)
from data_pipelines.tracing import (
    object_trace,
)


@flow(
    name="deep-search-tasks-generate-qas-for-entity",
    flow_run_name="entity-qa-{entity_index}-{entity_fact.entity_name}",
    retries=0,
    persist_result=False,
)
async def generate_deep_search_qas_for_entity_flow(
    entity_index: int,
    entity_fact: EntityFactMemoryRecord,
    description: str | None,
    paths: FinalizePaths,
    qa_pairs_per_entity: int,
    qa_model_id: str,
    complex_pair_ratio: float,
    max_qa_generation_attempts: int,
) -> EntityQASummary:
    target = qa_pairs_per_entity
    quota = ComplexityQuota.from_ratio(target, complex_pair_ratio)
    accepted = quota.accepted
    seen_questions: set[str] = set()
    errors: list[str] = []
    infrastructure_errors: list[Exception] = []
    with object_trace(
        f"deep-search-tasks-qa-{entity_index}-{entity_fact.entity_name}",
        entity_fact.to_dict(),
        {"entity.name": entity_fact.entity_name, "requested_qas": target},
        project_name=phoenix_project(),
    ) as root:
        for _ in range(max_qa_generation_attempts):
            if len(accepted) >= target:
                break
            targets = quota.generation_targets()
            outcomes = await asyncio.gather(
                *(
                    generate_qa_candidate(
                        qa_model_id,
                        entity_fact.entity_name,
                        entity_fact.facts,
                        description,
                        complex_target,
                        root.carrier,
                    )
                    for complex_target in targets
                ),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, asyncio.CancelledError):
                    raise outcome
                if isinstance(outcome, Exception):
                    infrastructure_errors.append(outcome)
                    errors.append(f"{type(outcome).__name__}: {outcome}")
                    continue
                candidate, error = outcome
                if error:
                    errors.append(error)
                if candidate is None:
                    continue
                key = " ".join(candidate.question.lower().split())
                if not key or key in seen_questions:
                    continue
                if not quota.can_accept(candidate):
                    continue
                accepted.append(candidate)
                seen_questions.add(key)
                await append_output_record(paths.qas, candidate.to_dict(), paths.lock)
                if len(accepted) >= target:
                    break
        if len(accepted) < target:
            await append_output_record(
                paths.failures,
                {
                    "stage": "qa_generation_shortfall",
                    "entity": entity_fact.entity_name,
                    "requested": target,
                    "generated": len(accepted),
                    "candidate_errors": errors,
                },
                paths.lock,
            )
        if len(accepted) < target and infrastructure_errors:
            raise RuntimeError(
                f"QA provider failed after {max_qa_generation_attempts} attempt rounds "
                f"for {entity_fact.entity_name!r}; generated {len(accepted)}/{target}"
            ) from infrastructure_errors[-1]
        root.set_output({"requested": target, "generated": len(accepted)})
        return EntityQASummary(
            entity_name=entity_fact.entity_name,
            requested=target,
            generated=len(accepted),
            phoenix_trace_id=root.trace_id,
        )


@task(name="append-entity-fact-record", retries=0, persist_result=False)
async def append_output_record(
    path: Path, value: dict[str, Any], lock_path: Path
) -> None:
    await anyio.to_thread.run_sync(append_jsonl, path, value, lock_path)
