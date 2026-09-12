import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from data_pipelines.artifacts.retrieval_queries import RetrievalChunk
from data_pipelines.pipelines.deep_search_task_generation.facts import (
    EntityFactMemoryRecord,
    ExtractedFact,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa import (
    candidates,
    entity,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.models import (
    ComplexityLevel,
    GeneratedQA,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.output import (
    initialize_finalize_output,
)
from data_pipelines.pipelines.search_query_generation import stages
from data_pipelines.pipelines.search_query_generation.config import (
    RetrievalQueriesConfig,
)
from data_pipelines.pipelines.search_query_generation.models import (
    FilterReason,
    QueryStatus,
    RetrievalQuery,
)
from data_pipelines.providers.openai import LLMCompletion
from ragent_core.retrievers.document import DOCUMENT_ID_KEY, Document, RetrievalResult

pytestmark = pytest.mark.usefixtures("local_tracing")


@asynccontextmanager
async def no_concurrency_service(*args, **kwargs):
    yield


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "narrow_output,expected_status",
    [
        (
            "<result><keep>true</keep><query>Specific query</query><hard_negatives><chunk id='1'/></hard_negatives></result>",
            QueryStatus.READY,
        ),
        ("<result><keep>false</keep></result>", QueryStatus.FILTERED),
        ("malformed", QueryStatus.FAILED),
    ],
)
async def test_search_generation_lifecycle(narrow_output, expected_status, monkeypatch):
    config = RetrievalQueriesConfig(table_name="source")
    outputs = iter(["<result><query>Query</query></result>", narrow_output])

    async def complete(*args):
        return LLMCompletion(next(outputs), "fake-model")

    retriever = SimpleNamespace(
        retrieve=lambda *args, **kwargs: [
            RetrievalResult(id=0, content="Positive", metadata={DOCUMENT_ID_KEY: 0}),
            RetrievalResult(id=1, content="Negative", metadata={DOCUMENT_ID_KEY: 1}),
        ]
    )
    monkeypatch.setattr(stages, "concurrency", no_concurrency_service)
    monkeypatch.setattr(stages, "chat_completion", complete)
    monkeypatch.setattr(
        stages,
        "chunk_by_source_index",
        lambda *args: Document(id=0, content="Positive", document_id=0),
    )
    monkeypatch.setattr(stages, "load_retriever", lambda _: retriever)
    query = await stages.sample_chunk.fn(config, 0, 0, {}, {})
    query = await stages.generate_query.fn(query, config, 0, 0, {})
    assert query.status is QueryStatus.GENERATED
    query = await stages.retrieve_candidates.fn(query, config, 0, 0, {})
    assert query.status is QueryStatus.MINED
    query = await stages.contrastive_narrow.fn(query, config, 0, 0, {})
    assert query.status is expected_status
    if expected_status is QueryStatus.READY:
        assert query.query == "Specific query"
        assert [chunk.id for chunk in query.hard_negatives] == [1]
        assert query.to_training_record(1)["positive"][DOCUMENT_ID_KEY] == 0
    elif expected_status is QueryStatus.FILTERED:
        assert query.reason_code is FilterReason.CONTRASTIVE_REJECTION
    else:
        assert query.failure_reason


@pytest.mark.asyncio
async def test_round_trip_miss_skips_contrastive_provider(monkeypatch):

    config = RetrievalQueriesConfig(table_name="source", round_trip_top_k=1)
    monkeypatch.setattr(stages, "concurrency", no_concurrency_service)
    monkeypatch.setattr(
        stages,
        "load_retriever",
        lambda _: SimpleNamespace(
            retrieve=lambda *args, **kwargs: [RetrievalResult(id=1)]
        ),
    )
    query = RetrievalQuery(
        "Query",
        0,
        positive=RetrievalChunk(id=0, document_id=0),
        status=QueryStatus.GENERATED,
    )
    filtered = await stages.retrieve_candidates.fn(query, config, 0, 0, {})
    assert filtered.reason_code is FilterReason.ROUND_TRIP_MISS
    assert await stages.contrastive_narrow.fn(filtered, config, 0, 0, {}) is filtered


@pytest.mark.asyncio
async def test_qa_retry_replaces_duplicates_and_meets_complexity_quota(
    tmp_path, monkeypatch
):
    facts = EntityFactMemoryRecord(
        "Entity", "source", (0, 1), (ExtractedFact("Fact", [0, 1]),)
    )
    paths = initialize_finalize_output(tmp_path, "run12345")
    attempts = []

    async def candidate(model, name, facts, description, complex_target, carrier):
        question = "Duplicate" if len(attempts) < 2 else "Unique"
        attempts.append(complex_target)
        return GeneratedQA(
            question=question,
            answer="Answer",
            doc_ids=[0, 1],
            complexity=ComplexityLevel.COMPLEX
            if complex_target
            else ComplexityLevel.SIMPLE,
        ), None

    monkeypatch.setattr(entity, "generate_qa_candidate", candidate)
    monkeypatch.setattr(entity, "append_output_record", entity.append_output_record.fn)
    result = await entity.generate_deep_search_qas_for_entity_flow.fn(
        0, facts, None, paths, 2, "fake", 0.5, 2
    )
    assert result.generated == 2
    assert attempts == [True, False, False]
    records = [json.loads(line) for line in paths.qas.read_text().splitlines()]
    assert [record["info"]["complexity"] for record in records] == [
        ComplexityLevel.COMPLEX.value,
        ComplexityLevel.SIMPLE.value,
    ]
    assert not paths.failures.read_text()


@pytest.mark.asyncio
async def test_qa_shortfall_is_persisted_after_exhausted_attempts(
    tmp_path, monkeypatch
):
    async def candidate(*args):
        return None, "provider response invalid"

    monkeypatch.setattr(entity, "generate_qa_candidate", candidate)
    monkeypatch.setattr(entity, "append_output_record", entity.append_output_record.fn)
    facts = EntityFactMemoryRecord(
        "Entity", "source", (0, 1), (ExtractedFact("Fact", [0, 1]),)
    )
    paths = initialize_finalize_output(tmp_path, "run12345")
    result = await entity.generate_deep_search_qas_for_entity_flow.fn(
        0, facts, None, paths, 1, "fake", 0.5, 2
    )
    assert result.generated == 0
    failure = json.loads(paths.failures.read_text())
    assert failure["requested"] == 1
    assert len(failure["candidate_errors"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output,accepted",
    [
        ("malformed", False),
        (
            "<qa><question>Q</question><answer>A</answer><doc_ids>0,999</doc_ids></qa>",
            False,
        ),
        (
            "<qa><question>Q</question><answer>A</answer><doc_ids>0,1,1</doc_ids></qa>",
            True,
        ),
    ],
)
async def test_qa_candidate_grounds_in_available_facts(output, accepted, monkeypatch):

    async def complete(*args):
        return LLMCompletion(output, "fake")

    monkeypatch.setattr(candidates, "concurrency", no_concurrency_service)
    monkeypatch.setattr(candidates, "chat_completion", complete)
    qa, error = await candidates.generate_qa_candidate.fn(
        "fake", "Entity", [ExtractedFact("Fact", [0, 1])], None, True, {}
    )
    assert (qa is not None) is accepted
    if accepted:
        assert qa.doc_ids == [0, 1]
        assert error is None
    else:
        assert error


@pytest.mark.asyncio
@pytest.mark.parametrize("recovers", [False, True])
async def test_qa_provider_failure_is_retried_but_never_reported_as_empty(
    tmp_path, monkeypatch, recovers
):
    calls = 0

    async def candidate(*args):
        nonlocal calls
        calls += 1
        if calls == 1 or not recovers:
            raise ConnectionError("provider unavailable")
        return GeneratedQA(
            question="Recovered",
            answer="Answer",
            doc_ids=[0, 1],
            complexity=ComplexityLevel.SIMPLE,
        ), None

    monkeypatch.setattr(entity, "generate_qa_candidate", candidate)
    monkeypatch.setattr(entity, "append_output_record", entity.append_output_record.fn)
    facts = EntityFactMemoryRecord(
        "Entity", "source", (0, 1), (ExtractedFact("Fact", [0, 1]),)
    )
    paths = initialize_finalize_output(tmp_path, "run12345")
    run = entity.generate_deep_search_qas_for_entity_flow.fn(
        0, facts, None, paths, 1, "fake", 0, 2
    )
    if recovers:
        assert (await run).generated == 1
    else:
        with pytest.raises(RuntimeError, match="QA provider failed"):
            await run
        failure = json.loads(paths.failures.read_text())
        assert len(failure["candidate_errors"]) == 2
    assert calls == 2
