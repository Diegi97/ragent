import json

import pytest

from data_pipelines.pipelines.deep_search_task_generation.facts import (
    EntityFactBatchRequest,
)
from data_pipelines.pipelines.deep_search_task_generation.generate import (
    GenerationStatus,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.fact_extraction_output.diagnostics import (
    ParseDiagnostics,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.fact_extraction_output.inputs import (
    load_batch_input_metadata,
    parse_batch_output_files,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.fact_extraction_output.joining import (
    build_entity_facts_from_batch_output,
)


def request_metadata(tmp_path):
    request = EntityFactBatchRequest(
        "request", "Entity", "corpus", (0,), (1,), "prompt"
    )
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(request.to_fireworks_record()) + "\n")
    return load_batch_input_metadata(path)


def response(custom_id, content):
    return {
        "custom_id": custom_id,
        "response": {"choices": [{"message": {"content": content}}]},
    }


def test_fact_provenance_rejects_entire_unsupported_fact(tmp_path):
    diagnostics = ParseDiagnostics()
    records = build_entity_facts_from_batch_output(
        {
            "request": "<facts><fact><statement>Supported</statement><doc_ids>0</doc_ids></fact><fact><statement>Fabricated</statement><doc_ids>0,99</doc_ids></fact></facts>"
        },
        request_metadata(tmp_path),
        diagnostics,
    )
    assert [fact.statement for fact in records[0].facts] == ["Supported"]
    assert records[0].facts[0].doc_ids == [0]
    assert diagnostics.unsupported_facts == 1
    assert diagnostics.has_integrity_errors


def test_identical_batch_replay_is_idempotent(tmp_path):
    path = tmp_path / "response.jsonl"
    record = response("request", "<facts></facts>")
    path.write_text("\n".join(json.dumps(record) for _ in range(2)))
    responses, diagnostics = parse_batch_output_files([path])
    records = build_entity_facts_from_batch_output(
        responses, request_metadata(tmp_path), diagnostics
    )
    assert len(records) == 1 and not records[0].facts
    assert diagnostics.duplicate_custom_ids == 1
    assert not diagnostics.has_integrity_errors


def test_conflicting_duplicate_cannot_restore_a_response(tmp_path):
    path = tmp_path / "response.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(response("request", value))
            for value in ["first", "second", "first"]
        )
    )
    responses, diagnostics = parse_batch_output_files([path])
    assert responses == {}
    assert diagnostics.conflicting_custom_ids == 1
    assert diagnostics.has_integrity_errors


def test_missing_and_unknown_responses_are_degraded(tmp_path):
    diagnostics = ParseDiagnostics()
    assert (
        build_entity_facts_from_batch_output(
            {"unknown": "<facts></facts>"}, request_metadata(tmp_path), diagnostics
        )
        == []
    )
    assert diagnostics.unmatched_responses == 1
    assert diagnostics.missing_responses == 1
    assert (
        GenerationStatus.for_output(2, 2, diagnostics.has_integrity_errors)
        is GenerationStatus.PARTIAL
    )


def test_metadata_rejects_duplicate_request_keys(tmp_path):
    request_metadata(tmp_path)
    path = tmp_path / "requests.jsonl"
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_batch_input_metadata(path)


def test_malformed_fact_blocks_are_not_a_valid_empty_response(tmp_path):
    diagnostics = ParseDiagnostics()
    assert (
        build_entity_facts_from_batch_output(
            {"request": "<facts><fact>unfinished</facts>"},
            request_metadata(tmp_path),
            diagnostics,
        )
        == []
    )
    assert diagnostics.invalid_contents == 1


def test_response_id_whitespace_matches_prepared_metadata(tmp_path):
    path = tmp_path / "response.jsonl"
    path.write_text(json.dumps(response(" request ", "<facts></facts>")))
    responses, diagnostics = parse_batch_output_files([path])
    records = build_entity_facts_from_batch_output(
        responses, request_metadata(tmp_path), diagnostics
    )
    assert len(records) == 1
    assert diagnostics.unmatched_responses == 0
    assert diagnostics.missing_responses == 0
