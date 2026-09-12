import asyncio
import logging
import signal

import pytest

from data_pipelines.pipelines.deep_search_task_generation.facts import (
    EntityFactMemoryRecord,
    ExtractedFact,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics import (
    attempts,
    runtime,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.audit_contract import (
    SolverAudit,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.config import (
    PiThinkingLevel,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.models import (
    QuestionRubricAssignment,
    QuestionRubricAttempt,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.output import (
    initialize_rubric_finalize_output,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    validate_question_rubric_file,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation.markdown import (
    example_markdown,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.workspace import (
    create_fact_workspace,
)


@pytest.mark.asyncio
async def test_duplicate_rubric_is_retried_and_accepted_audits_are_retained(
    tmp_path, monkeypatch
):
    entity = EntityFactMemoryRecord(
        "Entity", "source", (123, 456, 789), (ExtractedFact("Fact", [123, 456, 789]),)
    )
    paths = initialize_rubric_finalize_output(tmp_path, "run12345")
    workspace = create_fact_workspace(paths.workspace_directory, [entity])
    candidate = workspace.outputs_directory / "candidate.md"
    candidate.write_text(example_markdown())
    record = validate_question_rubric_file(candidate)
    assignments = [
        QuestionRubricAssignment(slot=slot, entity_fact=entity) for slot in range(2)
    ]
    calls = []

    async def attempt(assignment, **kwargs):
        calls.append((assignment.slot, kwargs["attempt"]))
        current = (
            record
            if kwargs["attempt"] == 1
            else record.model_copy(update={"question": "Distinct question"})
        )
        audit = SolverAudit(
            ok=True,
            candidate_sha256="hash",
            question=current.question,
            criteria_total=2,
            criteria_passed=1,
            percent_passed=50,
        )
        return QuestionRubricAttempt(
            assignment=assignment, record=current, solver_audit=audit
        )

    monkeypatch.setattr(attempts, "run_question_rubric_attempt", attempt)
    result = await attempts.generate_question_rubrics(
        assignments,
        model="fake",
        solver_model="fake",
        thinking=None,
        max_attempts=2,
        workspace=workspace,
        sessions_directory=paths.sessions_directory,
        paths=paths,
        pi_concurrency=1,
        logger=logging.getLogger(__name__),
    )
    assert len(result.accepted) == 2
    assert set(result.solver_audits) == set(result.accepted)
    assert sum(number == 2 for _, number in calls) == 1
    assert len(paths.question_rubrics.read_text().splitlines()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_pi_timeout_and_cancellation_stop_process_group(
    tmp_path, monkeypatch, cancel
):
    extension = tmp_path / runtime.PI_PHOENIX_EXTENSION
    extension.parent.mkdir(parents=True)
    extension.touch()
    monkeypatch.setenv(runtime.PI_CODING_AGENT_DIRECTORY_ENV, str(tmp_path))
    monkeypatch.setattr(runtime, "PI_TIMEOUT_SECONDS", 0.05)
    started = asyncio.Event()
    killed = []
    calls = []

    class Process:
        pid = 123456789
        returncode = None

        async def communicate(self):
            started.set()
            await asyncio.Event().wait()

        async def wait(self):
            self.returncode = -signal.SIGTERM
            return self.returncode

    async def launch(*command, **kwargs):
        assert command[command.index("--thinking") + 1] == PiThinkingLevel.HIGH.value
        calls.append(kwargs)
        return Process()

    monkeypatch.setattr(runtime.asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(
        runtime.os, "killpg", lambda pid, sig: killed.append((pid, sig))
    )
    task = asyncio.create_task(
        runtime.run_pi(
            prompt="prompt",
            model="fake",
            thinking=PiThinkingLevel.HIGH,
            working_directory=tmp_path,
            session_directory=tmp_path,
            session_name="test",
            system_instructions="instructions",
            environment={},
        )
    )
    await started.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert calls[0]["start_new_session"] is True
    assert killed == [(Process.pid, signal.SIGTERM)]


@pytest.mark.asyncio
async def test_exhausted_pi_execution_failure_raises(tmp_path, monkeypatch):
    entity = EntityFactMemoryRecord(
        "Entity", "source", (0,), (ExtractedFact("Fact", [0]),)
    )
    paths = initialize_rubric_finalize_output(tmp_path, "run12345")
    workspace = create_fact_workspace(paths.workspace_directory, [entity])
    assignment = QuestionRubricAssignment(slot=0, entity_fact=entity)

    async def failed_attempt(assignment, **kwargs):
        return QuestionRubricAttempt(
            assignment=assignment, error="PI unavailable", infrastructure_error=True
        )

    monkeypatch.setattr(attempts, "run_question_rubric_attempt", failed_attempt)
    with pytest.raises(RuntimeError, match="PI unavailable"):
        await attempts.generate_question_rubrics(
            [assignment],
            model="fake",
            solver_model="fake",
            thinking=None,
            max_attempts=1,
            workspace=workspace,
            sessions_directory=paths.sessions_directory,
            paths=paths,
            pi_concurrency=1,
            logger=logging.getLogger(__name__),
        )


@pytest.mark.asyncio
async def test_pi_launch_error_is_classified_as_infrastructure(
    tmp_path, monkeypatch, local_tracing
):
    entity = EntityFactMemoryRecord(
        "Entity", "source", (0,), (ExtractedFact("Fact", [0]),)
    )
    paths = initialize_rubric_finalize_output(tmp_path, "run12345")
    workspace = create_fact_workspace(paths.workspace_directory, [entity])
    assignment = QuestionRubricAssignment(slot=0, entity_fact=entity)

    async def unavailable(**kwargs):
        raise FileNotFoundError("PI executable missing")

    monkeypatch.setattr(attempts, "run_pi", unavailable)
    outcome = await attempts.run_question_rubric_attempt.fn(
        assignment,
        attempt=1,
        previous_errors=[],
        model="fake",
        solver_model="fake",
        thinking=None,
        workspace=workspace,
        sessions_directory=paths.sessions_directory,
    )
    assert outcome.record is None
    assert outcome.infrastructure_error
    assert "PI executable missing" in outcome.error
