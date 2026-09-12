import json
import logging
from types import SimpleNamespace

import pytest
from datasets import Dataset

from data_pipelines.pipelines.deep_search_task_generation.prepare import pipeline
from data_pipelines.pipelines.deep_search_task_generation.prepare.config import (
    DeepSearchTaskGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.entities import (
    load_entities_file,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.entity_matching import (
    EntityMatcher,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.models import (
    Concept,
    PrepareFailure,
    PrepareRunMetadata,
    PrepareStage,
    PrepareStatus,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.output import (
    initialize_prepare_output,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.requests import (
    FactRequestBuilder,
)
from data_pipelines.providers.fireworks import FireworksUploadResult
from ragent_core.retrievers.document import DOCUMENT_ID_KEY, RetrievalResult


def test_entity_file_deduplicates_names_and_checks_source_ids(tmp_path):
    path = tmp_path / "entities.jsonl"
    entities = [
        Concept("First", "source", 0),
        Concept(" FIRST ", "source", 0),
        Concept("Second", "source", 1),
    ]
    path.write_text("\n".join(json.dumps(entity.to_dict()) for entity in entities))
    assert [
        entity.name for entity in load_entities_file(path, "source", {0, 1}, 2)
    ] == ["First", "Second"]
    with pytest.raises(ValueError, match="unknown document"):
        load_entities_file(path, "source", {0}, 2)
    with pytest.raises(ValueError, match="data_source"):
        load_entities_file(path, "other", {0, 1}, 2)


def test_fact_request_preserves_parent_ids_and_links_entities():
    request = FactRequestBuilder(EntityMatcher(["First", "Second"])).build(
        0,
        0,
        Concept("First", "source", 0),
        [
            RetrievalResult(
                id=9,
                content="First works with Second",
                title="Title",
                metadata={DOCUMENT_ID_KEY: 0},
            )
        ],
        None,
    )
    assert request.doc_ids == (0,)
    assert request.chunk_ids == (9,)
    assert "Second" in request.prompt
    assert request.to_fireworks_record()["doc_ids"] == [0]


@pytest.mark.asyncio
async def test_prepare_discovery_isolates_failed_entity_requests(tmp_path, monkeypatch):
    path = tmp_path / "entities.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(Concept(name, "source", index).to_dict())
            for index, name in enumerate(["First", "Second"])
        )
    )
    config = DeepSearchTaskGenerationConfig(
        data_source="source", num_entities=2, entities_file=path, output_root=tmp_path
    )
    paths = initialize_prepare_output(config, "run12345")

    async def corpus(_):
        return Dataset.from_dict({"id": [0, 1]}), "source", None

    async def requests(client, config, index, *args):
        if index == 0:
            raise RuntimeError("retrieval unavailable")
        return []

    monkeypatch.setattr(pipeline, "load_corpus", corpus)
    monkeypatch.setattr(pipeline, "retrieve_and_prepare_entity_requests", requests)
    source, entities, outcomes = await pipeline.discover_entities_and_prepare_requests(
        config, SimpleNamespace(), paths, [], logging.getLogger(__name__)
    )
    assert source == "source"
    assert len(entities) == 2
    assert isinstance(outcomes[0], RuntimeError)
    assert outcomes[1] == []
    assert len(paths.entities.read_text().splitlines()) == 2


@pytest.fixture
def prepare_flow(tmp_path, monkeypatch):
    config = DeepSearchTaskGenerationConfig(
        data_source="source", num_entities=1, output_root=tmp_path
    )
    paths = initialize_prepare_output(config, "run12345")

    async def no_op(*args):
        return None

    async def initialize(*args):
        return paths

    class Worker:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def health(self):
            return SimpleNamespace(model_dump=lambda **kwargs: {})

    monkeypatch.setattr(pipeline, "get_run_logger", lambda: logging.getLogger(__name__))
    monkeypatch.setattr(
        pipeline, "flow_run", SimpleNamespace(id="00000000-0000-0000-0000-000000000001")
    )
    monkeypatch.setattr(
        pipeline,
        "configure_tracing",
        lambda **kwargs: SimpleNamespace(project_name="test", force_flush=lambda: None),
    )
    monkeypatch.setattr(pipeline, "_upsert_llm_concurrency_limit", no_op)
    monkeypatch.setattr(pipeline, "initialize_prepare", initialize)
    monkeypatch.setattr(pipeline, "AsyncRetrieverWorkerClient", Worker)
    return config, paths


@pytest.mark.asyncio
async def test_prepare_provider_failure_writes_failed_metadata(
    prepare_flow, monkeypatch
):
    config, paths = prepare_flow

    async def discover(config, client, paths, failures, logger):
        failures.append(
            PrepareFailure(
                stage=PrepareStage.ENTITY_EXTRACTION, error="provider unavailable"
            )
        )
        return "source", [], []

    monkeypatch.setattr(pipeline, "discover_entities_and_prepare_requests", discover)
    with pytest.raises(RuntimeError, match="Entity extraction failed"):
        await pipeline.prepare_deep_search_tasks_flow.fn(config)
    assert (
        json.loads(paths.metadata.read_text())["status"] == PrepareStatus.FAILED.value
    )


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"doc_id": True},
        {"doc_id": "0"},
        {"name": None},
        {"info": []},
        {"unexpected": "value"},
    ],
)
def test_entity_file_rejects_invalid_typed_records(tmp_path, invalid_fields):
    path = tmp_path / "entities.jsonl"
    payload = Concept("First", "source", 0).to_dict() | invalid_fields
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"Invalid entity record.*entities.jsonl:1"):
        load_entities_file(path, "source", {0, 1}, 2)


def test_duplicate_entity_record_still_requires_valid_source_identity(tmp_path):
    path = tmp_path / "entities.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(entity.to_dict())
            for entity in [Concept("First", "source", 0), Concept("First", "other", 0)]
        )
    )
    with pytest.raises(ValueError, match="data_source"):
        load_entities_file(path, "source", {0}, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("upload_succeeds", [True, False])
async def test_prepare_upload_outcome_persists_before_return_or_raise(
    prepare_flow, monkeypatch, upload_succeeds
):
    config, paths = prepare_flow
    entity = Concept("First", "source", 0)
    request = FactRequestBuilder(EntityMatcher(["First"])).build(
        0,
        0,
        entity,
        [RetrievalResult(id=9, content="First", metadata={DOCUMENT_ID_KEY: 0})],
        None,
    )

    async def discover(*args):
        return "source", [entity], [[request]]

    async def upload(path, dataset_name, timeout):
        assert path == paths.fact_requests
        if not upload_succeeds:
            raise RuntimeError("upload unavailable")
        return FireworksUploadResult(dataset_name, 1)

    monkeypatch.setattr(pipeline, "discover_entities_and_prepare_requests", discover)
    monkeypatch.setattr(pipeline, "upload_requests", upload)
    if upload_succeeds:
        await pipeline.prepare_deep_search_tasks_flow.fn(config)
    else:
        with pytest.raises(RuntimeError, match="Fireworks dataset upload failed"):
            await pipeline.prepare_deep_search_tasks_flow.fn(config)
    metadata = json.loads(paths.metadata.read_text())
    assert (
        metadata["status"]
        == (
            PrepareStatus.UPLOADED if upload_succeeds else PrepareStatus.UPLOAD_FAILED
        ).value
    )
    assert metadata["fact_request_count"] == 1
    if upload_succeeds:
        assert metadata["fireworks"]["upload_payload"] == {
            "dataset_name": metadata["fireworks"]["input_dataset_name"],
            "example_count": 1,
        }
    else:
        assert "upload unavailable" in paths.failures.read_text()


def test_prepare_metadata_retains_legacy_extensions_and_validates_fireworks():
    raw = {"config": {"data_source": "source"}, "historical_diagnostic": {"count": 1}}
    metadata = PrepareRunMetadata.model_validate(raw)
    assert metadata.config.data_source == "source"
    assert metadata.fireworks.input_dataset_name is None
    assert metadata.model_dump(exclude_unset=True)["historical_diagnostic"] == {
        "count": 1
    }
    with pytest.raises(ValueError, match="fireworks"):
        PrepareRunMetadata.model_validate({**raw, "fireworks": []})
