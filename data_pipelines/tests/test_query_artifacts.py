import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from data_pipelines.artifacts.retrieval_queries import RetrievalChunk, TrainingRecord
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.config import (
    RetrieverWorkerConfig,
)
from data_pipelines.pipelines.retrieval_evaluation.config import (
    RetrievalEvaluationConfig,
)
from data_pipelines.pipelines.retrieval_evaluation.contracts import (
    QueryEvaluationStatus,
)
from data_pipelines.pipelines.retrieval_evaluation.evaluator import evaluate_retrieval
from data_pipelines.pipelines.retrieval_evaluation.inputs import (
    load_dataset_context,
    load_query_records,
)
from data_pipelines.pipelines.retrieval_evaluation.metrics import (
    aggregate_metrics,
    metrics_for_rank,
)
from data_pipelines.pipelines.search_query_generation.config import (
    RetrievalQueriesConfig,
)
from data_pipelines.pipelines.search_query_generation.models import (
    FilterReason,
    QueryStatus,
    RetrievalQuery,
)
from data_pipelines.pipelines.search_query_generation.output import (
    append_query_record,
    initialize_output,
    write_metadata,
)
from ragent_core.retrievers.document import DOCUMENT_ID_KEY, RetrievalResult
from ragent_core.retrievers.mode import RetrievalMode
from ragent_core.retrievers.retriever import TurbopufferRetriever


def generated_run(tmp_path):
    config = RetrievalQueriesConfig(
        table_name="source", output_path=tmp_path / "custom.jsonl"
    )
    paths = initialize_output(config, "run12345")
    query = RetrievalQuery(
        "question",
        0,
        positive=RetrievalChunk(id=0, document_id=0),
        status=QueryStatus.READY,
    )
    append_query_record(query, paths, 0)
    append_query_record(replace(query, query="broken"), paths, 0)
    write_metadata(config, paths, "run12345", "now", "project", [])
    return paths


def test_relocated_custom_filename_round_trip(tmp_path):
    paths = generated_run(tmp_path)
    destination = tmp_path / "relocated"
    paths.output_directory.rename(destination)
    context = load_dataset_context(destination)
    assert context.queries_path == destination / "custom.jsonl"
    records = load_query_records(context.queries_path)
    assert records[0].positive_chunk_id == 0
    assert records[0].positive_document_id == 0


@pytest.mark.parametrize("bad_id", [True, [], {}, "", "   "])
def test_training_record_rejects_invalid_identifiers(bad_id):
    with pytest.raises(ValidationError):
        TrainingRecord.model_validate(
            {"query": "q", "positive": {"id": bad_id, DOCUMENT_ID_KEY: 0}}
        )


def test_filter_reason_serialization_and_reset():
    query = RetrievalQuery("q", 0).failed(
        QueryStatus.FILTERED, "top 50 missed", FilterReason.ROUND_TRIP_MISS
    )
    record = query.to_failure_record()
    assert record["reason_code"] == FilterReason.ROUND_TRIP_MISS.value
    assert record["failure_reason"] == "top 50 missed"
    reset = query.with_metadata(QueryStatus.GENERATED)
    assert reset.reason_code is None
    assert reset.failure_reason is None


def test_evaluator_preserves_successful_query_metrics_on_failure(tmp_path, monkeypatch):
    paths = generated_run(tmp_path)

    class Retriever:
        def retrieve(self, query, **kwargs):
            if query == "broken":
                raise RuntimeError("offline")
            hit = RetrievalResult(
                id=0, content="c", metadata={DOCUMENT_ID_KEY: 0}, score=1
            )
            return [hit, hit]

    monkeypatch.setattr(
        TurbopufferRetriever, "load_index", lambda **kwargs: Retriever()
    )
    config = RetrievalEvaluationConfig(
        input_directory=paths.output_directory,
        search_type=RetrievalMode.BM25,
        top_k=1,
        cutoffs=(1,),
    )
    summary = evaluate_retrieval(config)
    assert summary.successful_queries == 1
    assert summary.failed_queries == 1
    assert summary.coverage == 0.5
    details = [
        json.loads(line) for line in summary.details_path.read_text().splitlines()
    ]
    assert len(details[0]["retrieved_results"]) == 1
    assert details[1]["status"] == QueryEvaluationStatus.ERROR.value
    assert summary.metrics["chunk"]["cutoffs"]["1"]["recall"] == 1


def test_failed_query_is_routed_out_of_training_data(tmp_path):
    config = RetrievalQueriesConfig(
        table_name="source", output_path=tmp_path / "queries.jsonl"
    )
    paths = initialize_output(config, "failed-run")
    query = RetrievalQuery("q", 0).failed(
        QueryStatus.FILTERED, "positive missing", FilterReason.ROUND_TRIP_MISS
    )
    assert append_query_record(query, paths, 0) == paths.failures_path
    assert paths.output_path.read_text() == ""
    assert (
        json.loads(paths.failures_path.read_text())["reason_code"]
        == FilterReason.ROUND_TRIP_MISS.value
    )
    metadata = write_metadata(config, paths, "failed-run", "now", "project", [])
    assert metadata["failure_records"] == 1
    assert metadata["trainable_records"] == 0


def test_successful_misses_and_beyond_cutoff_hits_contribute_zero():
    for rank in (None, 5):
        assert all(value == 0 for value in metrics_for_rank(rank, (3,))["3"].values())
    metrics = aggregate_metrics([None, 5], (3,))
    assert all(value == 0 for value in metrics["cutoffs"]["3"].values())
    assert metrics["rank_statistics"]["misses"] == 1
    assert metrics["rank_statistics"]["hits"] == 1
    assert metrics["query_count"] == 2


@pytest.mark.parametrize("threshold", [float("inf"), float("-inf"), float("nan")])
def test_retrieval_config_rejects_nonfinite_thresholds(tmp_path, threshold):
    with pytest.raises(ValidationError):
        RetrievalEvaluationConfig(
            input_directory=tmp_path, reranker_threshold=threshold
        )
    with pytest.raises(ValidationError):
        RetrieverWorkerConfig(rerank_threshold=threshold)


@pytest.mark.parametrize("namespace", [None, " ", []])
def test_query_metadata_rejects_missing_or_invalid_namespace_before_loading_records(
    tmp_path, namespace
):
    paths = generated_run(tmp_path)
    raw = json.loads(paths.metadata_path.read_text())
    if namespace is None:
        raw["config"].pop("logical_namespace")
    else:
        raw["config"]["logical_namespace"] = namespace
    paths.metadata_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="logical_namespace"):
        load_dataset_context(paths.output_directory)
