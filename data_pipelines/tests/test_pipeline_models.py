from types import SimpleNamespace
from uuid import uuid4

import pytest

from data_pipelines.pipelines.deep_search_task_generation.generate.qa.models import (
    ComplexityLevel,
    ComplexityQuota,
    GeneratedQA,
)
from data_pipelines.providers.fireworks import (
    FIREWORKS_RESOURCE_ID_MAX_LENGTH,
    fact_dataset_name,
)
from data_pipelines.providers.openai import LLMCompletion, ModelResponseError
from ragent_core.artifacts.question_rubric import QuestionRubricRecord


def test_complexity_quota_reserves_required_slots_and_preserves_artifact():
    quota = ComplexityQuota.from_ratio(3, 0.5)
    simple = GeneratedQA(question="q", answer="a", complexity=ComplexityLevel.SIMPLE)
    complex_qa = GeneratedQA(
        question="q2", answer="a2", complexity=ComplexityLevel.COMPLEX
    )
    assert quota.generation_targets() == [True, True, False]
    quota.accepted.append(simple)
    assert not quota.can_accept(simple)
    assert quota.can_accept(complex_qa)
    assert simple.to_dict()["info"]["complexity"] == ComplexityLevel.SIMPLE.value


def test_fireworks_names_are_unique_and_within_provider_limits():
    first = fact_dataset_name("A long source_" * 20, str(uuid4()))
    second = fact_dataset_name("A long source_" * 20, str(uuid4()))
    assert first != second
    assert len(first) <= FIREWORKS_RESOURCE_ID_MAX_LENGTH
    assert first[0].isalpha() and first[-1].isalnum()
    assert all(
        character.islower() or character.isdigit() or character == "-"
        for character in first
    )


@pytest.mark.parametrize(
    "choices",
    [
        [],
        [SimpleNamespace(message=SimpleNamespace(content=None, refusal=None))],
        [SimpleNamespace(message=SimpleNamespace(content="", refusal=None))],
        [SimpleNamespace(message=SimpleNamespace(content="text", refusal="no"))],
    ],
)
def test_completion_adapter_rejects_missing_text_or_refusal(choices):
    with pytest.raises(ModelResponseError):
        LLMCompletion.from_response(
            SimpleNamespace(choices=choices, usage=None, model="model"), "model"
        )


def test_empty_fact_list_is_valid_completion_text():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="<facts></facts>", refusal=None)
            )
        ],
        usage=None,
        model="model",
    )
    assert LLMCompletion.from_response(response, "model").content == "<facts></facts>"


def test_shared_rubric_schema_keeps_legacy_labels_and_zero_document_id():
    record = QuestionRubricRecord.model_validate(
        {
            "entity": "Entity",
            "question": "Question",
            "rubric": [{"criterion": "Requirement", "doc_ids": [0]}],
            "doc_ids": [0],
            "evolution_strategies": ["Historical strategy"],
            "question_type": "legacy",
        },
        strict=True,
    )
    assert record.model_dump(mode="json")["evolution_strategies"] == [
        "Historical strategy"
    ]
    assert "question_type" not in record.model_dump()
