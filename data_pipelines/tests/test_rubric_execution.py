import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_pipelines.pipelines.deep_search_task_generation.facts import (
    EntityFactMemoryRecord,
    ExtractedFact,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics import (
    attempts,
    retrieval_probe,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.audit_contract import (
    AUDITS_DIRECTORY_ENV,
    AuditPaths,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.models import (
    QuestionRubricAssignment,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.output import (
    initialize_rubric_finalize_output,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.solver import (
    SolverRuntime,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.solver.config import (
    SolverSettings,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    validate_question_rubric_file,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation.audits import (
    question_rubric_sha256,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation.markdown import (
    example_markdown,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.workspace import (
    create_fact_workspace,
)
from ragent_core.judges.criteria import criterion_id, criterion_metric_name
from ragent_core.retrievers.tool_protocol import ToolName


@pytest.fixture
def candidate(tmp_path):
    path = tmp_path / "candidate.md"
    path.write_text(example_markdown())
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["easy", "hard", "provider_error"])
async def test_probe_persists_completed_decision_or_incomplete_audit(
    candidate, tmp_path, monkeypatch, outcome
):
    monkeypatch.setenv(AUDITS_DIRECTORY_ENV, str(tmp_path))
    record = validate_question_rubric_file(candidate)

    async def search(command, queries):
        assert command == ToolName.SEARCH
        assert queries == [record.question]
        if outcome == "provider_error":
            raise RuntimeError("retrieval unavailable")
        doc_ids = record.doc_ids if outcome == "easy" else record.doc_ids[:1]
        return (
            "<search_results>"
            + "".join(f"<result><id>{doc_id}</id></result>" for doc_id in doc_ids)
            + "</search_results>"
        )

    monkeypatch.setattr(retrieval_probe, "_call_tool", search)
    if outcome == "provider_error":
        with pytest.raises(RuntimeError, match="retrieval unavailable"):
            await retrieval_probe._run_probe(candidate)
    else:
        result = await retrieval_probe._run_probe(candidate)
        assert result["probe_passed"] is (outcome == "hard")
        assert result["too_easy"] is (outcome == "easy")
    persisted = json.loads(AuditPaths(tmp_path, candidate.name).retrieval.read_text())
    assert persisted["ok"] is (outcome != "provider_error")
    assert persisted["candidate_sha256"] == question_rubric_sha256(candidate)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["success", "missing_episode", "failed_episode", "failed_trace"]
)
async def test_solver_persists_audit_and_rejects_failed_rollouts(
    candidate, tmp_path, outcome
):
    audits_directory = tmp_path / "audits"
    audits_directory.mkdir()
    settings = SolverSettings(
        tmp_path / "evaluation.toml", "source", audits_directory, "fake"
    )
    settings.evaluation_config.touch()
    trace = SimpleNamespace(
        ok=outcome != "failed_trace",
        last_error=SimpleNamespace(message="rollout failure"),
        info={},
        metrics={
            criterion_metric_name(criterion_id(1)): 1.0,
            criterion_metric_name(criterion_id(2)): 0.0,
        },
        last_reply="answer [123]",
    )
    episode = SimpleNamespace(
        ok=outcome != "failed_episode",
        traces=[trace],
        last_error=SimpleNamespace(message="episode failure"),
    )

    async def run_eval(environment, config):
        assert config.push is False
        dataset_path = Path(config.env["taskset"]["dataset_path"])
        assert json.loads(dataset_path.read_text())["data_source"] == "source"
        return [] if outcome == "missing_episode" else [episode]

    runtime = SolverRuntime(
        settings,
        {},
        SimpleNamespace(load_environment=lambda config: object()),
        SimpleNamespace(model_validate=lambda raw: SimpleNamespace(**raw)),
        run_eval,
    )
    previous_directory = Path.cwd()
    if outcome == "success":
        result = await runtime.solve(candidate)
        assert result["percent_passed"] == 50
        assert result["criteria_total"] == 2
    else:
        with pytest.raises(RuntimeError):
            await runtime.solve(candidate)
    assert Path.cwd() == previous_directory
    persisted = json.loads(runtime.audit_path(candidate).read_text())
    assert persisted["ok"] is (outcome == "success")
    assert persisted["candidate_sha256"] == question_rubric_sha256(candidate)


@pytest.mark.asyncio
@pytest.mark.parametrize("valid_candidate", [True, False])
async def test_pi_completion_accepts_valid_audits_or_classifies_semantic_rejection(
    tmp_path, monkeypatch, local_tracing, candidate, valid_candidate
):
    record = validate_question_rubric_file(candidate)
    entity = EntityFactMemoryRecord(
        record.entity,
        "source",
        tuple(record.doc_ids),
        (ExtractedFact("Fact", record.doc_ids),),
    )
    paths = initialize_rubric_finalize_output(tmp_path, "run12345")
    workspace = create_fact_workspace(paths.workspace_directory, [entity])
    assignment = QuestionRubricAssignment(slot=0, entity_fact=entity)
    output = workspace.outputs_directory / assignment.filename

    async def complete(**kwargs):
        output.write_text(example_markdown() if valid_candidate else "invalid rubric")
        digest = question_rubric_sha256(output)
        audits = AuditPaths(workspace.audits_directory, output.name)
        audits.retrieval.write_text(
            json.dumps(
                {
                    "ok": True,
                    "candidate_sha256": digest,
                    "question": record.question,
                    "supporting_doc_ids": record.doc_ids,
                    "probe_passed": True,
                }
            )
        )
        audits.solver.write_text(
            json.dumps(
                {
                    "ok": True,
                    "candidate_sha256": digest,
                    "question": record.question,
                    "criteria_total": 2,
                    "judgments": [
                        {"id": criterion_id(1), "passed": True},
                        {"id": criterion_id(2), "passed": False},
                    ],
                    "criteria_passed": 1,
                    "percent_passed": 50,
                }
            )
        )

    monkeypatch.setattr(attempts, "run_pi", complete)
    result = await attempts.run_question_rubric_attempt.fn(
        assignment,
        attempt=1,
        previous_errors=[],
        model="fake",
        solver_model="fake",
        thinking=None,
        workspace=workspace,
        sessions_directory=paths.sessions_directory,
    )
    assert result.infrastructure_error is False
    if valid_candidate:
        assert result.record == record
        assert result.solver_audit.percent_passed == 50
        assert result.error is None
    else:
        assert result.record is None
        assert result.error
