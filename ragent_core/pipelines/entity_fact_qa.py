from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import backoff
from datasets import Dataset
from openai import AsyncOpenAI
from sklearn.model_selection import train_test_split
from tqdm.asyncio import tqdm

from ragent_core.config import OPENROUTER_API_KEY, OPENROUTER_URL
from ragent_core.data_sources import load_corpus
from ragent_core.pipelines.base import BasePipeline, PipelineMetadata
from ragent_core.pipelines.explorer_agent.gemini_client import GeminiClient
from ragent_core.pipelines.prompts.atomic_prompts import (
    CONCEPT_EXTRACTOR_PROMPT,
    format_prompt_with_description,
    parse_concepts,
)
from ragent_core.pipelines.cross_concept_synthesis import _NoOpRetriever
from ragent_core.pipelines.prompts.entity_fact_prompts import (
    FACT_EXTRACTION_PROMPT,
    FACT_REFINEMENT_PROMPT,
    FACT_TO_QA_PROMPT,
    ExtractedFact,
    parse_extracted_facts,
    parse_fact_grounded_qas,
)
from ragent_core.retrievers.bm25_retriever import BM25Retriever
from ragent_core.retrievers.hybrid_retriever import HybridRetriever
from ragent_core.types import QA, Concept

logger = logging.getLogger(__name__)

CONTENT_COLUMN = "text"
DEFAULT_CONCEPT_MODEL_ID = "gemini-2.5-flash"
DEFAULT_FACT_MODEL_ID = "gemini-2.5-flash-lite"
DEFAULT_QA_MODEL_ID = "gemini-3-flash-preview"
DEFAULT_LLM_CLIENT = "gemini"
DEFAULT_NUM_ENTITIES = 20
DEFAULT_QA_PAIRS_PER_ENTITY = 4
DEFAULT_FACT_REFINEMENT_CHUNK_SIZE = 3
DEFAULT_COMPLEX_PAIR_RATIO = 0.7
DEFAULT_MAX_QA_GENERATION_ATTEMPTS = 4
DEFAULT_MAX_CONCURRENT_REQUESTS = 25
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_MAX_DOCS_PER_ENTITY = 30
DEFAULT_SEED = 42
DEFAULT_ENABLE_TRAIN_EVAL_SPLIT = False
DEFAULT_TRAIN_SIZE = 0.9
DEFAULT_EVAL_SIZE = 0.1


@dataclass
class EntityFactMemoryRecord:
    """Persistable entity memory record produced before QA generation."""

    entity_name: str
    data_source: str
    concept_doc_id: int
    concept_importance: str
    entity_doc_ids: list[int]
    facts: list[ExtractedFact]


@dataclass
class EntityFactQAConfig:
    data_source: str
    llm_client: str = DEFAULT_LLM_CLIENT
    num_entities: int = DEFAULT_NUM_ENTITIES
    qa_pairs_per_entity: int = DEFAULT_QA_PAIRS_PER_ENTITY
    concept_model_id: str = DEFAULT_CONCEPT_MODEL_ID
    fact_model_id: str = DEFAULT_FACT_MODEL_ID
    qa_model_id: str = DEFAULT_QA_MODEL_ID
    seed: int = DEFAULT_SEED
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    fact_refinement_chunk_size: int = DEFAULT_FACT_REFINEMENT_CHUNK_SIZE
    complex_pair_ratio: float = DEFAULT_COMPLEX_PAIR_RATIO
    max_qa_generation_attempts: int = DEFAULT_MAX_QA_GENERATION_ATTEMPTS
    max_docs_per_entity: int = DEFAULT_MAX_DOCS_PER_ENTITY
    enable_train_eval_split: bool = DEFAULT_ENABLE_TRAIN_EVAL_SPLIT
    train_size: float | int | None = DEFAULT_TRAIN_SIZE
    eval_size: float | int | None = DEFAULT_EVAL_SIZE
    use_bm25: bool = False
    use_lite_retriever: bool = False


class EntityFactQAPipeline(BasePipeline):
    Config = EntityFactQAConfig
    metadata = PipelineMetadata(
        name="qa/entity_fact",
        description=(
            "Pipeline for entity-centric fact extraction with iterative refinement "
            "and multi-document QA generation."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._llm: Optional[Any] = None

    def _init_llm_client(self, config: EntityFactQAConfig, retriever: Any) -> None:
        client_name = config.llm_client.strip().lower()
        if client_name == "gemini":
            self._llm = GeminiClient(retriever)
            return
        if client_name == "openai":
            self._llm = _OpenRouterOpenAIClient()
            return
        raise ValueError(f"Unsupported llm_client: {config.llm_client}")

    async def generate(self, config: EntityFactQAConfig) -> list[QA]:
        entity_facts, data_source_description = await self.generate_entity_facts(config)
        qas = await self.generate_qas_from_entity_facts(
            config=config,
            entity_facts=entity_facts,
            data_source_description=data_source_description,
        )
        logger.info(
            "Entity fact pipeline complete: generated %d QA pairs across %d entities",
            len(qas),
            len(entity_facts),
        )
        return qas

    async def generate_entity_facts(
        self, config: EntityFactQAConfig
    ) -> tuple[list[EntityFactMemoryRecord], Optional[str]]:
        self._validate_config(config)
        dataset, retriever, semaphore, rng, data_source_description = (
            self._build_generation_context(config)
        )

        concepts = await self._generate_concepts(
            dataset=dataset,
            retriever=retriever,
            num_entities=config.num_entities,
            model_id=config.concept_model_id,
            rng=rng,
            semaphore=semaphore,
            data_source_description=data_source_description,
        )

        if not concepts:
            logger.warning("No concepts/entities generated; returning empty output.")
            return [], data_source_description

        tasks = [
            self._extract_entity_facts(
                concept=concept,
                retriever=retriever,
                config=config,
                semaphore=semaphore,
                data_source_description=data_source_description,
            )
            for concept in concepts
        ]
        entity_results = await tqdm.gather(
            *tasks,
            desc="Extracting entity facts",
            total=len(tasks),
        )
        entity_facts = [record for record in entity_results if record is not None]

        logger.info(
            "Entity fact stage complete: retained %d entities with extracted facts",
            len(entity_facts),
        )
        return entity_facts, data_source_description

    async def generate_qas_from_entity_facts(
        self,
        config: EntityFactQAConfig,
        entity_facts: Sequence[EntityFactMemoryRecord],
        data_source_description: Optional[str] = None,
    ) -> list[QA]:
        self._validate_config(config)
        if not entity_facts:
            logger.warning(
                "No entity facts provided for QA generation; returning empty output."
            )
            return []

        if self._llm is None or data_source_description is None:
            dataset, dataset_name, data_source_description = self._load_dataset(
                config.data_source
            )
            retriever = _NoOpRetriever()
            self._init_llm_client(config, retriever)

        semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        tasks = [
            self._generate_qas_for_entity_memory(
                entity_fact=entity_fact,
                config=config,
                semaphore=semaphore,
                data_source_description=data_source_description,
            )
            for entity_fact in entity_facts
        ]
        entity_results = await tqdm.gather(
            *tasks,
            desc="Generating QAs",
            total=len(tasks),
        )
        return [qa for qa_list in entity_results for qa in qa_list]

    async def _extract_entity_facts(
        self,
        concept: Concept,
        retriever: Any,
        config: EntityFactQAConfig,
        semaphore: asyncio.Semaphore,
        data_source_description: Optional[str],
    ) -> Optional[EntityFactMemoryRecord]:
        try:
            entity_doc_ids = self._retrieve_entity_doc_ids(
                retriever,
                concept,
                max_docs_per_entity=config.max_docs_per_entity,
            )
            if not entity_doc_ids:
                logger.warning("No documents retrieved for entity '%s'", concept.name)
                return None

            facts = await self._extract_facts_with_refinement(
                concept_name=concept.name,
                retriever=retriever,
                doc_ids=entity_doc_ids,
                model_id=config.fact_model_id,
                semaphore=semaphore,
                chunk_size=config.fact_refinement_chunk_size,
                data_source_description=data_source_description,
            )
            if not facts:
                logger.warning("No facts extracted for entity '%s'", concept.name)
                return None

            # Filter entities with insufficient facts or document coverage
            if len(facts) < 3:
                logger.info(
                    "Skipping entity '%s': only %d facts (minimum 3 required)",
                    concept.name,
                    len(facts),
                )
                return None

            unique_doc_ids = {doc_id for fact in facts for doc_id in fact.doc_ids}
            if len(unique_doc_ids) < 5:
                logger.info(
                    "Skipping entity '%s': only %d unique documents (minimum 5 required)",
                    concept.name,
                    len(unique_doc_ids),
                )
                return None

            return EntityFactMemoryRecord(
                entity_name=concept.name,
                data_source=concept.data_source,
                concept_doc_id=int(concept.doc_id),
                concept_importance=str(concept.importance or ""),
                entity_doc_ids=entity_doc_ids,
                facts=facts,
            )
        except Exception as exc:
            logger.warning("Entity processing failed for '%s': %s", concept.name, exc)
            return None

    async def _generate_qas_for_entity_memory(
        self,
        entity_fact: EntityFactMemoryRecord,
        config: EntityFactQAConfig,
        semaphore: asyncio.Semaphore,
        data_source_description: Optional[str],
    ) -> list[QA]:
        concept = Concept(
            name=entity_fact.entity_name,
            data_source=entity_fact.data_source,
            doc_id=int(entity_fact.concept_doc_id),
            importance=entity_fact.concept_importance,
        )
        return await self._generate_qas_from_facts(
            concept=concept,
            facts=list(entity_fact.facts),
            target_pairs=config.qa_pairs_per_entity,
            model_id=config.qa_model_id,
            semaphore=semaphore,
            complex_pair_ratio=config.complex_pair_ratio,
            max_attempts=config.max_qa_generation_attempts,
            data_source_description=data_source_description,
        )

    def _validate_config(self, config: EntityFactQAConfig) -> None:
        if config.num_entities <= 0:
            raise ValueError("num_entities must be positive")
        if config.qa_pairs_per_entity <= 0:
            raise ValueError("qa_pairs_per_entity must be positive")
        if config.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        if config.fact_refinement_chunk_size <= 0:
            raise ValueError("fact_refinement_chunk_size must be positive")
        if not (0.0 <= config.complex_pair_ratio <= 1.0):
            raise ValueError("complex_pair_ratio must be between 0.0 and 1.0")
        if config.max_qa_generation_attempts <= 0:
            raise ValueError("max_qa_generation_attempts must be positive")
        if config.max_docs_per_entity is not None and config.max_docs_per_entity <= 0:
            raise ValueError("max_docs_per_entity must be positive when provided")

    def _load_dataset(
        self, data_source: str
    ) -> tuple[Dataset, Optional[str], Optional[str]]:
        logger.info("=" * 80)
        logger.info("Starting entity fact QA pipeline")
        logger.info("Data source: %s", data_source)
        logger.info("=" * 80)

        dataset, name, description = load_corpus(data_source)
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected Dataset but got {type(dataset).__name__}")
        if CONTENT_COLUMN not in dataset.column_names:
            raise KeyError(
                f"Dataset '{data_source}' must include a '{CONTENT_COLUMN}' column."
            )

        logger.info("Loaded dataset with %d documents", len(dataset))
        return dataset, name, description

    def _build_generation_context(
        self, config: EntityFactQAConfig
    ) -> tuple[Dataset, Any, asyncio.Semaphore, random.Random, Optional[str]]:
        dataset, dataset_name, data_source_description = self._load_dataset(
            config.data_source
        )
        retriever = self._init_retriever(config, dataset, dataset_name)
        self._init_llm_client(config, retriever)
        semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        rng = random.Random(config.seed)
        return dataset, retriever, semaphore, rng, data_source_description

    def _init_retriever(
        self,
        config: EntityFactQAConfig,
        dataset: Dataset,
        dataset_name: Optional[str],
    ) -> Any:
        if config.use_bm25:
            logger.info("Using BM25Retriever")
            return BM25Retriever(dataset=dataset, dataset_name=dataset_name)

        hybrid_kwargs: dict[str, Any] = {}
        if config.use_lite_retriever:
            hybrid_kwargs["colbert_model_name"] = (
                "mixedbread-ai/mxbai-edge-colbert-v0-32m"
            )
            hybrid_kwargs["reranker_model_name"] = (
                "mixedbread-ai/mxbai-rerank-xsmall-v1"
            )
            hybrid_kwargs["rerank_threshold"] = 0.0
            logger.info("Using lightweight HybridRetriever configuration")
        else:
            logger.info("Using standard HybridRetriever configuration")

        return HybridRetriever(
            dataset=dataset,
            dataset_name=dataset_name,
            **hybrid_kwargs,
            device='cpu'
        )

    async def _generate_concepts(
        self,
        dataset: Dataset,
        retriever: Any,
        num_entities: int,
        model_id: str,
        rng: random.Random,
        semaphore: asyncio.Semaphore,
        data_source_description: Optional[str],
    ) -> list[Concept]:
        if self._llm is None:
            raise RuntimeError("Gemini client not initialized")

        concepts: list[Concept] = []
        seen_names: set[str] = set()
        sample_size = min(DEFAULT_SAMPLE_SIZE, len(dataset))
        max_iterations = max(30, num_entities * 2)
        no_progress_limit = 5
        no_progress_count = 0
        iteration = 0

        while len(concepts) < num_entities and iteration < max_iterations:
            iteration += 1
            indices = self._sample_indices(rng, len(dataset), sample_size)
            subset = dataset.select(indices)
            prompt_documents = self._format_documents(
                subset["text"],
                doc_ids=subset["id"],
                titles=subset["title"],
            )
            prompt = CONCEPT_EXTRACTOR_PROMPT.format(DOCUMENTS=prompt_documents)
            prompt = format_prompt_with_description(prompt, data_source_description)

            try:
                response = await self._llm.chat_completion_with_retry(
                    content=prompt,
                    model=model_id,
                    semaphore=semaphore,
                    use_tools=False,
                )
            except Exception as exc:
                logger.warning(
                    "Concept extraction failed on iteration %d: %s", iteration, exc
                )
                continue

            parsed = parse_concepts(response, dataset.info.dataset_name or "unknown")
            added = 0
            for concept in parsed:
                key = concept.name.strip().lower()
                if not key or key in seen_names:
                    continue
                concept_doc_id = int(concept.doc_id)
                # In case llm hallucinates document ID
                if not self._document_exists(retriever, concept_doc_id):
                    continue
                concept.doc_id = concept_doc_id
                seen_names.add(key)
                concepts.append(concept)
                added += 1
                if len(concepts) >= num_entities:
                    break

            if added == 0:
                no_progress_count += 1
                if no_progress_count >= no_progress_limit:
                    logger.warning(
                        "Stopping concept generation after %d no-progress iterations.",
                        no_progress_count,
                    )
                    break
            else:
                no_progress_count = 0

            logger.info(
                "[Concept iteration %d] +%d concepts (total %d/%d)",
                iteration,
                added,
                len(concepts),
                num_entities,
            )

        return concepts[:num_entities]

    def _retrieve_entity_doc_ids(
        self,
        retriever: Any,
        concept: Concept,
        max_docs_per_entity: int | None,
    ) -> list[int]:
        if isinstance(retriever, HybridRetriever):
            # top_k=None means "all documents above reranker threshold"
            retrieval_results = retriever.retrieve(
                concept.name,
                top_k=None,
                rerank_batch_size=8,
            )
        else:
            top_k = len(retriever.documents)
            retrieval_results = retriever.retrieve(concept.name, top_k=top_k)

        doc_ids: list[int] = []
        seen: set[int] = set()

        concept_doc_id = int(concept.doc_id)
        if concept_doc_id not in seen:
            seen.add(concept_doc_id)
            doc_ids.append(concept_doc_id)

        for result in retrieval_results:
            doc_id = int(result.doc_id)
            if doc_id in seen:
                continue
            if not self._document_exists(retriever, doc_id):
                continue
            seen.add(doc_id)
            doc_ids.append(doc_id)

        if max_docs_per_entity is not None:
            return doc_ids[:max_docs_per_entity]
        return doc_ids

    async def _extract_facts_with_refinement(
        self,
        concept_name: str,
        retriever: Any,
        doc_ids: list[int],
        model_id: str,
        semaphore: asyncio.Semaphore,
        chunk_size: int,
        data_source_description: Optional[str],
    ) -> list[ExtractedFact]:
        if self._llm is None:
            raise RuntimeError("Gemini client not initialized")

        chunks = list(self._batched(doc_ids, chunk_size))
        if not chunks:
            return []

        current_facts: list[ExtractedFact] = []
        processed_doc_ids: list[int] = []

        for chunk_idx, chunk_doc_ids in enumerate(chunks):
            chunk_doc_ids, chunk_doc_texts, chunk_doc_titles = (
                self._collect_document_texts(
                    retriever,
                    chunk_doc_ids,
                )
            )
            if not chunk_doc_ids:
                logger.warning(
                    "No valid documents found for '%s' in chunk %d",
                    concept_name,
                    chunk_idx,
                )
                continue
            for doc_id in chunk_doc_ids:
                if doc_id not in processed_doc_ids:
                    processed_doc_ids.append(doc_id)

            formatted_docs = self._format_documents(
                chunk_doc_texts,
                doc_ids=chunk_doc_ids,
                titles=chunk_doc_titles,
            )
            if chunk_idx == 0 or not current_facts:
                prompt = FACT_EXTRACTION_PROMPT.format(
                    ENTITY=concept_name,
                    DOCUMENTS=formatted_docs,
                )
            else:
                prompt = FACT_REFINEMENT_PROMPT.format(
                    ENTITY=concept_name,
                    CURRENT_FACTS=self._format_facts(current_facts),
                    ADDITIONAL_DOCUMENTS=formatted_docs,
                )

            prompt = format_prompt_with_description(prompt, data_source_description)
            try:
                response = await self._llm.chat_completion_with_retry(
                    content=prompt,
                    model=model_id,
                    semaphore=semaphore,
                    use_tools=False,
                )
                print("#"*80)
                print(response)
                print("#"*80)
            except Exception as exc:
                logger.warning(
                    "Fact extraction/refinement failed for '%s' on chunk %d: %s",
                    concept_name,
                    chunk_idx,
                    exc,
                )
                continue

            parsed_facts = parse_extracted_facts(response)
            filtered_facts = self._filter_facts(
                parsed_facts, allowed_doc_ids=processed_doc_ids
            )
            if not filtered_facts:
                logger.warning(
                    "No parseable facts for '%s' on chunk %d for concept '%s' in doc_ids %s; keeping previous facts.",
                    concept_name,
                    chunk_idx,
                    concept_name,
                    chunk_doc_ids,
                )
                continue

            if chunk_idx == 0 or not current_facts:
                # First chunk: assign sequential fact IDs.
                current_facts = [
                    ExtractedFact(
                        statement=f.statement,
                        doc_ids=f.doc_ids,
                        fact_id=i + 1,
                    )
                    for i, f in enumerate(filtered_facts)
                ]
            else:
                # Subsequent chunks: merge new/updated facts into the existing list.
                current_facts = self._merge_facts(current_facts, filtered_facts)

            logger.debug(
                "Fact refinement step %d for '%s' produced %d total facts",
                chunk_idx,
                concept_name,
                len(current_facts),
            )

        return current_facts

    async def _generate_qas_from_facts(
        self,
        concept: Concept,
        facts: list[ExtractedFact],
        target_pairs: int,
        model_id: str,
        semaphore: asyncio.Semaphore,
        complex_pair_ratio: float,
        max_attempts: int,
        data_source_description: Optional[str],
    ) -> list[QA]:
        if self._llm is None:
            raise RuntimeError("Gemini client not initialized")

        allowed_doc_ids = sorted({did for fact in facts for did in fact.doc_ids})
        if not allowed_doc_ids:
            return []
        allowed_doc_id_set = set(allowed_doc_ids)

        required_complex = min(
            target_pairs, math.ceil(target_pairs * complex_pair_ratio)
        )

        accepted: list[QA] = []
        seen_questions: set[str] = set()
        facts_block = self._format_facts(facts)

        async def _generate_single_candidate(must_be_complex: bool) -> Optional[QA]:
            prompt = FACT_TO_QA_PROMPT.format(
                ENTITY=concept.name,
                FACTS=facts_block,
                COMPLEXITY_TARGET="complex" if must_be_complex else "simple",
            )
            prompt_with_context = format_prompt_with_description(
                prompt, data_source_description
            )

            try:
                response = await self._llm.chat_completion_with_retry(
                    content=prompt_with_context,
                    model=model_id,
                    semaphore=semaphore,
                    use_tools=False,
                )
            except Exception as exc:
                logger.warning(
                    "QA generation failed for entity '%s': %s",
                    concept.name,
                    exc,
                )
                return None

            parsed_qas = parse_fact_grounded_qas(response)
            if not parsed_qas:
                return None

            candidate = parsed_qas[0]
            normalized_doc_ids = self._normalize_candidate_doc_ids(
                candidate.doc_ids,
                allowed_doc_ids=allowed_doc_id_set,
            )
            # Require at least 2 documents for multi-hop reasoning
            if len(normalized_doc_ids) < 2:
                return None

            question = candidate.question.strip()
            answer = candidate.answer.strip()
            if not question or not answer:
                return None

            return QA(
                question=question,
                answer=answer,
                doc_ids=normalized_doc_ids,
                info={
                    "entity": concept.name,
                    "complexity": "complex" if must_be_complex else "simple",
                    "num_docs": len(normalized_doc_ids),
                    "num_facts_for_entity": len(facts),
                    "facts": [
                        {
                            "fact_id": fact.fact_id,
                            "statement": fact.statement,
                            "doc_ids": fact.doc_ids,
                        }
                        for fact in facts
                    ],
                },
            )

        for _ in range(max_attempts):
            if len(accepted) >= target_pairs:
                break

            remaining_slots = target_pairs - len(accepted)
            current_complex = sum(
                1 for qa in accepted if qa.info.get("complexity") == "complex"
            )
            remaining_complex = max(0, required_complex - current_complex)
            complexity_targets = [True] * min(remaining_complex, remaining_slots)
            complexity_targets.extend(
                [False] * (remaining_slots - len(complexity_targets))
            )

            tasks = [
                _generate_single_candidate(must_be_complex=must_be_complex)
                for must_be_complex in complexity_targets
            ]
            round_candidates = await asyncio.gather(*tasks)

            for candidate in round_candidates:
                if candidate is None:
                    continue
                question_key = " ".join(candidate.question.lower().split())
                if not question_key or question_key in seen_questions:
                    continue

                # Keep room for required complex pairs.
                current_complex = sum(
                    1 for qa in accepted if qa.info.get("complexity") == "complex"
                )
                remaining_complex = max(0, required_complex - current_complex)
                remaining_slots = target_pairs - len(accepted)
                is_complex = candidate.info.get("complexity") == "complex"
                if not is_complex and remaining_slots <= remaining_complex:
                    continue

                accepted.append(candidate)
                seen_questions.add(question_key)
                if len(accepted) >= target_pairs:
                    break

        if len(accepted) < target_pairs:
            logger.warning(
                "Entity '%s': generated %d/%d QA pairs",
                concept.name,
                len(accepted),
                target_pairs,
            )

        complex_count = sum(
            1 for qa in accepted if qa.info.get("complexity") == "complex"
        )
        if complex_count < required_complex:
            logger.warning(
                "Entity '%s': generated %d/%d required complex QA pairs",
                concept.name,
                complex_count,
                required_complex,
            )

        return accepted[:target_pairs]

    def _normalize_candidate_doc_ids(
        self,
        doc_ids: Sequence[int],
        allowed_doc_ids: set[int],
    ) -> list[int]:
        normalized: list[int] = []
        seen: set[int] = set()
        for raw_doc_id in doc_ids:
            candidate = int(raw_doc_id)
            if candidate not in allowed_doc_ids:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
        return normalized

    def _document_exists(self, retriever: Any, doc_id: int) -> bool:
        documents = retriever.documents
        return doc_id in documents or str(doc_id) in documents

    def _get_document_by_id(
        self, retriever: Any, doc_id: int
    ) -> Optional[dict[str, Any]]:
        documents = retriever.documents
        document = documents.get(doc_id)
        if document is not None:
            return document
        return documents.get(str(doc_id))

    def _collect_document_texts(
        self,
        retriever: Any,
        doc_ids: Sequence[int],
    ) -> tuple[list[int], list[str], list[str]]:
        valid_doc_ids: list[int] = []
        doc_titles: list[str] = []
        doc_texts: list[str] = []
        for raw_doc_id in doc_ids:
            doc_id = int(raw_doc_id)
            document = self._get_document_by_id(retriever, doc_id)
            if document is None:
                continue
            text = str(document.get(CONTENT_COLUMN, "") or "")
            if not text:
                continue
            valid_doc_ids.append(doc_id)
            doc_texts.append(text)
            doc_titles.append(document.get("title", ""))
        return valid_doc_ids, doc_texts, doc_titles

    @staticmethod
    def _sample_indices(
        rng: random.Random,
        population: int,
        sample_size: int,
    ) -> list[int]:
        if population <= 0:
            return []
        k = max(1, min(sample_size, population))
        return rng.sample(range(population), k=k)

    @staticmethod
    def _batched(iterable: Iterable[int], size: int) -> Iterable[list[int]]:
        if size <= 0:
            raise ValueError("batch size must be positive")
        batch: list[int] = []
        for value in iterable:
            batch.append(value)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def _format_documents(
        documents: Iterable[str],
        titles: Iterable[str],
        doc_ids: Iterable[int],
    ) -> str:
        formatted = []
        for doc_id, text, title in zip(doc_ids, documents, titles):
            text_str = str(text)
            formatted.append(
                "<document>\n"
                f"<doc_id>{doc_id}</doc_id>\n"
                f"<title>{title}</title>\n"
                f"<content>{text_str}</content>\n"
                "</document>\n"
            )
        return "".join(formatted)

    @staticmethod
    def _format_facts(facts: Sequence[ExtractedFact]) -> str:
        lines = []
        for fact in facts:
            lines.append("<fact>")
            if fact.fact_id > 0:
                lines.append(f"<fact_id>{fact.fact_id}</fact_id>")
            lines.append(f"<statement>{fact.statement}</statement>")
            lines.append(f"<doc_ids>{','.join(str(d) for d in fact.doc_ids)}</doc_ids>")
            lines.append("</fact>")
        return "\n".join(lines)

    @staticmethod
    def _filter_facts(
        facts: Sequence[ExtractedFact],
        allowed_doc_ids: Sequence[int],
    ) -> list[ExtractedFact]:
        """Filter and deduplicate extracted facts based on document IDs and content.

        Removes facts that reference disallowed document IDs, have empty statements,
        or are duplicates (case-insensitive). Normalizes whitespace in statements
        before deduplication.
        """
        allowed = set(allowed_doc_ids)
        filtered: list[ExtractedFact] = []
        seen_statements: set[str] = set()

        for fact in facts:
            valid_ids = [d for d in fact.doc_ids if d in allowed]
            if not valid_ids:
                continue
            statement = " ".join(fact.statement.split()).strip()
            if not statement:
                continue
            key = statement.lower()
            if key in seen_statements:
                continue
            seen_statements.add(key)
            filtered.append(
                ExtractedFact(
                    statement=statement, doc_ids=valid_ids, fact_id=fact.fact_id
                )
            )
        return filtered

    @staticmethod
    def _merge_facts(
        existing: Sequence[ExtractedFact],
        new_facts: Sequence[ExtractedFact],
    ) -> list[ExtractedFact]:
        """Merge new facts into the existing fact list using fact_id.

        - If a new fact carries a fact_id that matches an existing entry,
          the existing entry is updated (statement replaced, doc_ids merged).
        - Otherwise the new fact is appended with the next available ID.
        """
        by_id: dict[int, int] = {}  # fact_id -> index in result
        result: list[ExtractedFact] = list(existing)

        for idx, fact in enumerate(result):
            if fact.fact_id > 0:
                by_id[fact.fact_id] = idx

        next_id = max((f.fact_id for f in result), default=0) + 1

        for fact in new_facts:
            if fact.fact_id > 0 and fact.fact_id in by_id:
                # Existing fact_id — correction or additional doc support.
                idx = by_id[fact.fact_id]
                old = result[idx]
                result[idx] = ExtractedFact(
                    statement=fact.statement,
                    doc_ids=sorted(set(old.doc_ids) | set(fact.doc_ids)),
                    fact_id=fact.fact_id,
                )
            else:
                # New fact — assign next ID and append.
                new_entry = ExtractedFact(
                    statement=fact.statement,
                    doc_ids=fact.doc_ids,
                    fact_id=next_id,
                )
                result.append(new_entry)
                by_id[next_id] = len(result) - 1
                next_id += 1

        return result

    def split_train_eval(
        self,
        qas: Sequence[QA],
        *,
        eval_size: float | int | None = None,
        train_size: float | int | None = None,
        seed: Optional[int] = None,
        stratify: bool = True,
        shuffle: bool = True,
    ) -> tuple[list[QA], list[QA]]:
        qas_list = list(qas)
        labels = [qa.info.get("complexity", "unknown") for qa in qas_list]
        if stratify:
            counts: dict[str, int] = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            if any(count < 2 for count in counts.values()):
                stratify = False
        stratify_labels: Optional[Sequence[str]] = labels if stratify else None

        train_qas, eval_qas = train_test_split(
            qas_list,
            test_size=eval_size,
            train_size=train_size,
            random_state=seed,
            shuffle=shuffle,
            stratify=stratify_labels,
        )
        return list(train_qas), list(eval_qas)


class _OpenRouterOpenAIClient:
    """Minimal OpenAI-compatible client for OpenRouter chat completions."""

    def __init__(self, client: Optional[AsyncOpenAI] = None) -> None:
        self._client = client or AsyncOpenAI(
            base_url=OPENROUTER_URL,
            api_key=OPENROUTER_API_KEY,
        )

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter
    )
    async def chat_completion_with_retry(
        self,
        content: str,
        model: str,
        semaphore: asyncio.Semaphore,
        system_prompt: Optional[str] = None,
        use_tools: bool = False,
    ) -> str:
        del use_tools  # Tool calling is not used by this pipeline.
        messages: list[dict[str, str]] = [{"role": "user", "content": content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        async with semaphore:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
            )

        return response.choices[0].message.content or ""


PIPELINE = EntityFactQAPipeline
