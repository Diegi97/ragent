from collections.abc import Sequence
from dataclasses import asdict

from prefect import task
from prefect.concurrency.asyncio import concurrency

from data_pipelines.pipelines.deep_search_task_generation.facts import ExtractedFact
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.models import (
    ComplexityLevel,
    GeneratedQA,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.prompts import (
    FACT_TO_QA_PROMPT,
    format_facts,
    parse_fact_grounded_qas,
)
from data_pipelines.pipelines.deep_search_task_generation.project import (
    LLM_CONCURRENCY_LIMIT,
    phoenix_project,
)
from data_pipelines.pipelines.deep_search_task_generation.prompts import (
    format_prompt_with_description,
)
from data_pipelines.providers.openai import chat_completion
from data_pipelines.tracing import (
    SpanKind,
    set_span_error,
    set_span_output,
    stage_span,
)


def normalize_candidate_doc_ids(
    doc_ids: Sequence[int], allowed_doc_ids: set[int]
) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for value in doc_ids:
        candidate = int(value)
        if candidate in allowed_doc_ids and candidate not in seen:
            normalized.append(candidate)
            seen.add(candidate)
    return normalized


@task(name="generate-entity-qa-candidate", retries=0, persist_result=False)
async def generate_qa_candidate(
    qa_model_id: str,
    entity_name: str,
    facts: Sequence[ExtractedFact],
    description: str | None,
    must_be_complex: bool,
    trace_carrier: dict[str, str],
) -> tuple[GeneratedQA | None, str | None]:
    complexity = ComplexityLevel.COMPLEX if must_be_complex else ComplexityLevel.SIMPLE
    prompt = FACT_TO_QA_PROMPT.format(
        ENTITY=entity_name,
        FACTS=format_facts(facts),
        COMPLEXITY_TARGET=complexity.value,
    )
    with stage_span(
        trace_carrier,
        "generate_qa_candidate",
        SpanKind.LLM,
        {"entity": entity_name, "complex": must_be_complex},
        {"llm.model_name": qa_model_id},
        project_name=phoenix_project(),
    ) as span:
        try:
            async with concurrency(LLM_CONCURRENCY_LIMIT, strict=True):
                response = await chat_completion(
                    [
                        {
                            "role": "user",
                            "content": format_prompt_with_description(
                                prompt, description
                            ),
                        }
                    ],
                    qa_model_id,
                )
            parsed = parse_fact_grounded_qas(response.content)
            if not parsed:
                reason = "Model response contained no valid QA pair."
                set_span_error(span, reason)
                return None, reason
            candidate = parsed[0]
            allowed = {doc_id for fact in facts for doc_id in fact.doc_ids}
            doc_ids = normalize_candidate_doc_ids(candidate.doc_ids, allowed)
            if len(doc_ids) < 2:
                reason = "QA pair is not grounded in at least two allowed documents."
                set_span_error(span, reason)
                return None, reason
            qa = GeneratedQA(
                complexity=complexity,
                question=candidate.question.strip(),
                answer=candidate.answer.strip(),
                doc_ids=doc_ids,
                info={
                    "entity": entity_name,
                    "num_docs": len(doc_ids),
                    "num_facts_for_entity": len(facts),
                    "facts": [asdict(fact) for fact in facts],
                },
            )
            if not qa.question or not qa.answer:
                reason = "QA pair contains an empty question or answer."
                set_span_error(span, reason)
                return None, reason
            set_span_output(span, qa.to_dict())
            return qa, None
        except Exception as exc:
            set_span_error(span, f"{type(exc).__name__}: {exc}")
            raise
