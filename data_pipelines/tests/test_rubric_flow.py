import json
import logging
from types import SimpleNamespace

import pytest

from data_pipelines.pipelines.deep_search_task_generation.facts import (
    EntityFactMemoryRecord,
    ExtractedFact,
)
from data_pipelines.pipelines.deep_search_task_generation.generate import (
    GenerationStatus,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.fact_extraction_output.diagnostics import (
    ParseDiagnostics,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics import (
    pipeline,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.audit_contract import (
    SolverAudit,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.models import (
    RubricGenerationResult,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.models import (
    PreparePaths,
)
from ragent_core.artifacts.question_rubric import (
    QuestionRubricDatasetMetadata,
    QuestionRubricRecord,
)

pytestmark = pytest.mark.usefixtures("local_tracing")


@pytest.mark.asyncio
@pytest.mark.parametrize("generation_fails", [False, True])
async def test_rubric_flow_persists_success_or_failure_artifacts(
    tmp_path, monkeypatch, generation_fails
):
    paths = PreparePaths.in_directory(tmp_path)
    paths.metadata.write_text(
        json.dumps(
            {
                "config": {"data_source": "source"},
                "prefect_flow_run_id": "prepare-run",
                "fireworks": {"input_dataset_name": "input-dataset"},
            }
        )
    )
    paths.fact_responses.touch()
    paths.entities.write_text(json.dumps({"name": "Entity"}) + "\n")
    entity = EntityFactMemoryRecord(
        "Entity", "source", (0, 1), (ExtractedFact("Fact", [0, 1]),)
    )
    record = QuestionRubricRecord(
        entity="Entity",
        question="Question",
        rubric=[
            {"criterion": "First", "doc_ids": [0]},
            {"criterion": "Second", "doc_ids": [1]},
        ],
        doc_ids=[0, 1],
    )
    generation_calls = []

    async def parse(*args):
        return [{}], [entity], ParseDiagnostics()

    async def generate(assignments, **kwargs):
        generation_calls.append(assignments)
        assert kwargs["workspace"].allowed_doc_ids == frozenset({0, 1})
        if generation_fails:
            raise RuntimeError("Pi unavailable")
        audit = SolverAudit(
            ok=True,
            candidate_sha256="hash",
            question=record.question,
            criteria_total=2,
            criteria_passed=1,
            percent_passed=50,
        )
        return RubricGenerationResult(
            accepted={0: record},
            solver_audits={0: audit},
            phoenix_trace_ids={0: "phoenix-trace"},
        )

    monkeypatch.setattr(pipeline, "flow_run", SimpleNamespace(id="generation-run"))
    monkeypatch.setattr(pipeline, "get_run_logger", lambda: logging.getLogger(__name__))
    monkeypatch.setattr(pipeline, "parse_fact_output", parse)
    monkeypatch.setattr(pipeline, "generate_question_rubrics", generate)
    if generation_fails:
        with pytest.raises(RuntimeError, match="generate_question_rubrics"):
            await pipeline.generate_deep_search_rubrics_flow.fn(tmp_path, "fake", 1)
    else:
        returned = await pipeline.generate_deep_search_rubrics_flow.fn(
            tmp_path, "fake", 1
        )
    (metadata_path,) = tmp_path.glob("rubric_finalize_*/metadata.json")
    metadata = json.loads(metadata_path.read_text())
    failures = [
        json.loads(line)
        for line in metadata_path.with_name("failures.jsonl").read_text().splitlines()
    ]
    assert len(generation_calls) == 1
    assert generation_calls[0][0].entity_fact == entity
    assert metadata["prepare_prefect_flow_run_id"] == "prepare-run"
    assert metadata["fireworks"]["input_dataset_name"] == "input-dataset"
    if generation_fails:
        assert metadata["status"] == GenerationStatus.FAILED
        assert metadata["failed_stage"] == "generate_question_rubrics"
        assert failures[0]["error"] == "RuntimeError: Pi unavailable"
    else:
        assert metadata == returned
        assert metadata["status"] == GenerationStatus.COMPLETED
        assert metadata["question_rubric_count"] == 1
        assert (
            QuestionRubricDatasetMetadata.model_validate(
                metadata
            ).prepare_config.data_source
            == "source"
        )
        assert json.loads(
            metadata_path.with_name("question_rubrics.jsonl").read_text()
        ) == record.model_dump(mode="json")
        assert failures == []
