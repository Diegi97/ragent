from collections.abc import Sequence
from pathlib import Path

import anyio
from prefect import task

from data_pipelines.pipelines.deep_search_task_generation.facts import (
    EntityFactBatchRequest,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.config import (
    DeepSearchTaskGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.entity_matching import (
    EntityMatcher,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.models import Concept
from data_pipelines.pipelines.deep_search_task_generation.prepare.output import (
    write_retrieval_debug,
)
from data_pipelines.pipelines.deep_search_task_generation.project import (
    phoenix_project,
)
from data_pipelines.pipelines.deep_search_task_generation.prompts import (
    format_prompt_with_description,
)
from data_pipelines.pipelines.deep_search_task_generation.prompts.facts import (
    FACT_EXTRACTION_PROMPT,
    format_documents,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.client import (
    AsyncRetrieverWorkerClient,
)
from data_pipelines.tracing import (
    SpanKind,
    object_trace,
    set_span_output,
    stage_span,
)
from ragent_core.retrievers.document import RetrievalResult


class FactRequestBuilder:
    def __init__(self, entity_matcher: EntityMatcher) -> None:
        self._entity_matcher = entity_matcher

    def prepare(
        self,
        config: DeepSearchTaskGenerationConfig,
        entity_index: int,
        entity: Concept,
        description: str | None,
        retrieval_debug_directory: Path,
        chunks: Sequence[RetrievalResult],
    ) -> list[EntityFactBatchRequest]:
        ordered_chunks = list(chunks)
        write_retrieval_debug(
            retrieval_debug_directory / f"entity-{entity_index:05d}.json",
            entity.name,
            config.num_chunks_per_entity,
            ordered_chunks,
        )
        ordered_chunks.sort(key=lambda chunk: int(chunk.id))
        requests: list[EntityFactBatchRequest] = []
        size = config.fact_extraction_chunks_per_request
        for group_index, start in enumerate(range(0, len(ordered_chunks), size)):
            group = ordered_chunks[start : start + size]
            requests.append(
                self.build(entity_index, group_index, entity, group, description)
            )
        return requests

    def build(
        self,
        entity_index: int,
        group_index: int,
        entity: Concept,
        group: Sequence[RetrievalResult],
        description: str | None,
    ) -> EntityFactBatchRequest:
        doc_ids = tuple(chunk.source_document_id for chunk in group)
        titles = [chunk.title for chunk in group]
        combined_text = "\n".join(chunk.content for chunk in group)
        matched = set(self._entity_matcher.match(combined_text))
        linked = "\n".join(
            candidate_name
            for candidate_name in self._entity_matcher.entities
            if candidate_name != entity.name and candidate_name in matched
        )
        prompt = FACT_EXTRACTION_PROMPT.format(
            ENTITY=entity.name,
            ENTITIES=linked,
            PASSAGE=format_documents(
                [chunk.content for chunk in group], titles, doc_ids
            ),
        )
        return EntityFactBatchRequest(
            key=f"entity-{entity_index:05d}-group-{group_index:05d}",
            entity_name=entity.name,
            data_source=entity.data_source,
            doc_ids=doc_ids,
            chunk_ids=tuple(int(chunk.id) for chunk in group),
            prompt=format_prompt_with_description(prompt, description),
        )


async def retrieve_and_prepare_entity_requests(
    retriever_client: AsyncRetrieverWorkerClient,
    config: DeepSearchTaskGenerationConfig,
    entity_index: int,
    entity: Concept,
    request_builder: FactRequestBuilder,
    description: str | None,
    table_name: str,
    retrieval_debug_directory: Path,
) -> list[EntityFactBatchRequest]:
    chunks = await retriever_client.retrieve(
        entity.name,
        table_name=table_name,
        top_k=config.num_chunks_per_entity,
    )
    return await prepare_entity_requests(
        config,
        entity_index,
        entity,
        request_builder,
        description,
        retrieval_debug_directory,
        chunks,
    )


@task(
    name="prepare-fact-extraction-requests",
    task_run_name="prepare-facts-{entity_index}",
    retries=0,
    persist_result=False,
)
async def prepare_entity_requests(
    config: DeepSearchTaskGenerationConfig,
    entity_index: int,
    entity: Concept,
    request_builder: FactRequestBuilder,
    description: str | None,
    retrieval_debug_directory: Path,
    chunks: Sequence[RetrievalResult],
) -> list[EntityFactBatchRequest]:
    with object_trace(
        f"fact-request-preparation-{entity_index}-{entity.name}",
        entity.to_dict(),
        {"entity.name": entity.name},
        project_name=phoenix_project(),
    ) as root:
        with stage_span(
            root.carrier,
            "prepare_fact_requests",
            SpanKind.CHAIN,
            entity.to_dict(),
            project_name=phoenix_project(),
        ) as span:
            requests = await anyio.to_thread.run_sync(
                request_builder.prepare,
                config,
                entity_index,
                entity,
                description,
                retrieval_debug_directory,
                chunks,
            )
            set_span_output(span, {"request_count": len(requests)})
        root.set_output({"request_count": len(requests)})
        return requests
