from __future__ import annotations

import asyncio
import inspect
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import backoff
from datasets import Dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.auto import tqdm
from tqdm.asyncio import tqdm_asyncio

from ragent_core.config import OPENROUTER_API_KEY, OPENROUTER_URL
from ragent_core.data_sources import load_corpus
from ragent_core.pipelines.base import BasePipeline, PipelineMetadata

from ragent_core.pipelines.prompts.atomic_prompts import (
    CONCEPT_EXTRACTOR_PROMPT,
    parse_concepts,
    ANSWER_GENERATION_PROMPT,
    ANSWER_REFINMENT_PROMPT,
    QUESTION_GENERATION_PROMPT,
    extract_batch_qas_from_text,
    format_prompt_with_description,
)
from ragent_core.retrievers.bm25_retriever import BM25Retriever
from ragent_core.retrievers.hybrid_retriever import HybridRetriever
from ragent_core.types import Concept, QA

logger = logging.getLogger(__name__)

CONTENT_COLUMN = "text"
DEFAULT_QUESTION_MODEL_ID = "google/gemini-2.5-flash"
DEFAULT_ANSWER_MODEL_ID = "google/gemini-2.5-flash"
DEFAULT_NUM_PAIRS = 10
DEFAULT_CHUNK_SIZE = 3
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_RANDOM_SEED = 43
DEFAULT_MAX_CONCURRENT_REQUESTS = 100
DEFAULT_MAX_CONCEPTS_PER_DOC_CALL = 5
DEFAULT_MAX_ANSWER_DOCS = 15

load_dotenv()


DEFAULT_CHECKPOINT_INTERVAL = 10


@dataclass
class AtomicQAConfig:
    data_source: str
    refinement_chunk_size: int = DEFAULT_CHUNK_SIZE
    use_minimal_bm25: bool = False
    num_pairs: int = DEFAULT_NUM_PAIRS
    question_model_id: str = DEFAULT_QUESTION_MODEL_ID
    answer_model_id: str = DEFAULT_ANSWER_MODEL_ID
    seed: int = DEFAULT_RANDOM_SEED
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    max_concepts_per_doc: int = DEFAULT_MAX_CONCEPTS_PER_DOC_CALL
    max_answer_docs: int = DEFAULT_MAX_ANSWER_DOCS
    enable_train_eval_split: bool = False
    # Checkpoint configuration
    checkpoint_dir: str | None = None
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    resume_from: str | None = None
    enable_checkpointing: bool = True


class AtomicQAPipeline(BasePipeline):
    """Reference QA generation pipeline built around concept extraction and refinement."""

    Config = AtomicQAConfig
    metadata = PipelineMetadata(
        name="qa/atomic",
        description="Pipeline for generating atomic and simple question answer pairs from a data source.",
    )

    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        super().__init__()
        self._client = client or AsyncOpenAI(
            base_url=OPENROUTER_URL,
            api_key=OPENROUTER_API_KEY,
        )

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter
    )
    async def _chat_completion_with_retry(
        self,
        model: str,
        messages: list[dict[str, Any]],
        semaphore: asyncio.Semaphore,
        postprocess: Optional[Callable[[Any], Any | Awaitable[Any]]] = None,
    ) -> Optional[Any]:
        """Invoke the chat completion API with optional post-processing and retries."""
        async with semaphore:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
            )
            response_text = response.choices[0].message.content
            if postprocess is None:
                return response_text

            processed_text = postprocess(response_text)
            if inspect.isawaitable(processed_text):
                processed_text = await processed_text
            return processed_text

    async def generate(self, config: AtomicQAConfig) -> list[QA]:
        if config.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")

        semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        # Initialize checkpointing
        resumed_qas: list[QA] = []
        resumed_state: dict = {}
        if config.enable_checkpointing:
            self._init_checkpoint_manager(
                checkpoint_dir=config.checkpoint_dir,
                checkpoint_interval=config.checkpoint_interval,
            )

        # Handle resume from checkpoint
        if config.resume_from:
            from ragent_core.pipelines.base import CheckpointManager

            resumed_qas, resumed_state, resume_metadata = (
                CheckpointManager.load_checkpoint(config.resume_from)
            )
            logger.info(
                "Resuming from checkpoint with %d existing QA pairs",
                len(resumed_qas),
            )

        dataset, dataset_name, data_source_description = self._load_dataset(
            config.data_source
        )
        question_model = config.question_model_id
        answer_model = config.answer_model_id

        rng = random.Random(config.seed)
        logger.info(
            "Configuration: refinement_chunk_size=%d, use_minimal_bm25=%s, "
            "num_pairs=%d, question_model_id=%s, "
            "answer_model_id=%s, max_concurrent_requests=%d, "
            "max_concepts_per_doc=%d, max_answer_docs=%d",
            config.refinement_chunk_size,
            config.use_minimal_bm25,
            config.num_pairs,
            question_model,
            answer_model,
            config.max_concurrent_requests,
            config.max_concepts_per_doc,
            config.max_answer_docs,
        )

        # Calculate remaining pairs needed
        remaining_pairs = max(0, config.num_pairs - len(resumed_qas))
        if remaining_pairs == 0:
            logger.info(
                "Already have %d QA pairs from checkpoint, no more needed",
                len(resumed_qas),
            )
            return resumed_qas

        concepts = await self._generate_concepts(
            dataset,
            num_pairs=remaining_pairs,
            question_model_id=question_model,
            rng=rng,
            semaphore=semaphore,
            data_source_description=data_source_description,
        )

        rng.shuffle(
            concepts
        )  # So that the first concepts are not about the same document

        if remaining_pairs is not None:
            concepts = concepts[:remaining_pairs]

        qas = await self._generate_questions(
            concepts,
            dataset,
            question_model_id=question_model,
            max_concepts_per_doc=config.max_concepts_per_doc,
            semaphore=semaphore,
            data_source_description=data_source_description,
        )
        self._retrieve_supporting_documents_for_qas(
            qas,
            dataset=dataset,
            dataset_name=dataset_name,
            use_minimal_bm25=config.use_minimal_bm25,
            max_answer_docs=config.max_answer_docs,
        )

        answered_qas = await self._generate_answers(
            qas,
            dataset=dataset,
            refinement_model_id=answer_model,
            refinement_chunk_size=config.refinement_chunk_size,
            semaphore=semaphore,
            existing_qas=resumed_qas,
            data_source_description=data_source_description,
        )

        # Save final checkpoint
        self._save_final_checkpoint(answered_qas)

        logger.info("=" * 80)
        logger.info("✓ QA generation pipeline complete!")
        logger.info("Final output: %d QA pairs", len(answered_qas))
        logger.info("=" * 80)

        return answered_qas

    def _load_dataset(
        self, data_source: str
    ) -> tuple[Dataset, Optional[str], Optional[str]]:
        logger.info("=" * 80)
        logger.info("Starting QA generation pipeline")
        logger.info("Data source: %s", data_source)
        logger.info("=" * 80)

        dataset, name, description = load_corpus(data_source)

        logger.info("Loaded dataset with %d documents", len(dataset))

        if CONTENT_COLUMN not in dataset.column_names:
            raise KeyError(
                f"Dataset '{data_source}' must include a '{CONTENT_COLUMN}' column after preprocessing."
            )

        return dataset, name, description

    # --------------------------------------------------------------------- #
    # Concept generation
    # --------------------------------------------------------------------- #
    async def _generate_concepts(
        self,
        dataset: Dataset,
        num_pairs: int,
        question_model_id: str,
        rng: random.Random,
        semaphore: asyncio.Semaphore,
        data_source_description: Optional[str] = None,
    ) -> list[Concept]:
        logger.info(
            "Generating at least %d concepts from dataset of %d documents",
            num_pairs,
            len(dataset),
        )

        if len(dataset) == 0:
            raise ValueError("Cannot generate concepts from an empty dataset")

        concepts: list[Concept] = []
        iteration = 0
        sample_size = min(DEFAULT_SAMPLE_SIZE, len(dataset))

        while len(concepts) < num_pairs:
            iteration += 1
            indices = self._sample_indices(rng, len(dataset), sample_size)
            subset = dataset.select(indices)

            prompt_documents = self._format_documents(
                subset[CONTENT_COLUMN], indices=indices
            )
            prompt = CONCEPT_EXTRACTOR_PROMPT.format(DOCUMENTS=prompt_documents)
            prompt = format_prompt_with_description(prompt, data_source_description)

            logger.debug(
                "Concept iteration %d: calling model with documents %s",
                iteration,
                indices,
            )

            def _parse(response: str) -> list[Concept]:
                parsed_concepts = parse_concepts(
                    response,
                    dataset.info.dataset_name or "unknown",
                )
                return parsed_concepts

            parsed = await self._chat_completion_with_retry(
                model=question_model_id,
                messages=[{"role": "user", "content": prompt}],
                semaphore=semaphore,
                postprocess=_parse,
            )
            if parsed is None:
                continue

            concepts.extend(parsed)
            logger.info(
                "[Iteration %d] +%d concepts (total %d/%d) from documents %s",
                iteration,
                len(parsed),
                len(concepts),
                num_pairs,
                indices,
            )

        logger.info("Concept generation complete: %d concepts", len(concepts))
        return concepts

    @staticmethod
    def _sample_indices(
        rng: random.Random, population: int, sample_size: int
    ) -> list[int]:
        sample_size = max(1, min(sample_size, population))
        return rng.sample(range(population), k=sample_size)

    @staticmethod
    def _format_documents(documents: Iterable[str], indices: Iterable[int]) -> str:
        formatted = []
        for idx, text in zip(indices, documents):
            formatted.append(
                f"<document>\n<doc_idx>{idx}</doc_idx>\n<content>{text}</content>\n</document>\n"
            )
        return "".join(formatted)

    @staticmethod
    def _format_concepts_for_prompt(concepts: Iterable[Concept]) -> str:
        formatted = []
        for concept in concepts:
            formatted.append(
                "<concept>\n"
                f"  <name>{concept.name}</name>\n"
                f"  <importance>{concept.importance}</importance>\n"
                "</concept>\n"
            )
        return "".join(formatted)

    @staticmethod
    def _chunk_concepts(concepts: list[Concept], size: int) -> Iterable[list[Concept]]:
        if size <= 0:
            raise ValueError("max_concepts_per_doc must be positive")
        for start in range(0, len(concepts), size):
            yield concepts[start : start + size]

    # --------------------------------------------------------------------- #
    # Question and answer generation
    # --------------------------------------------------------------------- #
    async def _generate_questions(
        self,
        concepts: list[Concept],
        dataset: Dataset,
        question_model_id: str,
        max_concepts_per_doc: int,
        semaphore: asyncio.Semaphore,
        data_source_description: Optional[str] = None,
    ) -> list[QA]:
        if max_concepts_per_doc <= 0:
            raise ValueError("max_concepts_per_doc must be positive")

        logger.info(
            "Generating questions for %d concepts (max concepts per doc call=%d)",
            len(concepts),
            max_concepts_per_doc,
        )

        if not concepts:
            logger.info("No concepts provided; skipping question generation")
            return []

        concepts_by_doc: dict[int, list[Concept]] = {}
        for concept in concepts:
            concepts_by_doc.setdefault(concept.doc_idx, []).append(concept)

        tasks = []
        batch_sizes = []

        async def _generate_batch(
            doc_idx: int, concept_batch: list[Concept], batch_number: int
        ) -> Optional[list[QA]]:
            source_doc = dataset[doc_idx]
            expected_concepts = list(concept_batch)
            prompt = QUESTION_GENERATION_PROMPT.format(
                SOURCE_DOCUMENT=str(source_doc[CONTENT_COLUMN]),
                CONCEPTS=self._format_concepts_for_prompt(expected_concepts),
            )
            prompt = format_prompt_with_description(prompt, data_source_description)

            def _parse(response: str) -> list[QA]:
                return extract_batch_qas_from_text(
                    response,
                    doc_idx=doc_idx,
                    concepts=expected_concepts,
                )

            return await self._chat_completion_with_retry(
                model=question_model_id,
                messages=[{"role": "user", "content": prompt}],
                semaphore=semaphore,
                postprocess=_parse,
            )

        for doc_idx, doc_concepts in concepts_by_doc.items():
            batches = list(self._chunk_concepts(doc_concepts, max_concepts_per_doc))
            for batch_number, concept_batch in enumerate(batches, start=1):
                tasks.append(_generate_batch(doc_idx, concept_batch, batch_number))
                batch_sizes.append(len(concept_batch))

        responses = await tqdm_asyncio.gather(
            *tasks,
            total=len(tasks),
            desc="Generating questions",
            unit="batch",
        )

        qas: list[QA] = []
        skipped_concepts = 0
        for batch_size, response in zip(batch_sizes, responses):
            if response is None:
                skipped_concepts += batch_size
                continue
            qas.extend(response)

        logger.info(
            "Generated %d QA skeletons (skipped %d concepts)",
            len(qas),
            skipped_concepts,
        )
        return qas

    async def _generate_answers(
        self,
        qas: list[QA],
        dataset: Dataset,
        refinement_model_id: str,
        refinement_chunk_size: int,
        semaphore: asyncio.Semaphore,
        existing_qas: list[QA] | None = None,
        data_source_description: Optional[str] = None,
    ) -> list[QA]:
        """Generate answers with incremental checkpointing support."""
        if refinement_chunk_size <= 0:
            raise ValueError("refinement_chunk_size must be positive")

        logger.info(
            "Generating and refining answers for %d questions (chunk size=%d)",
            len(qas),
            refinement_chunk_size,
        )

        answered_qas: list[QA] = list(existing_qas) if existing_qas else []

        # Create tasks for all QAs
        async def _process_qa(qa: QA) -> Optional[QA]:
            return await self._generate_answer_with_refinement(
                qa,
                dataset=dataset,
                model_id=refinement_model_id,
                refinement_chunk_size=refinement_chunk_size,
                semaphore=semaphore,
                data_source_description=data_source_description,
            )

        # Process QAs and checkpoint incrementally
        progress = tqdm(total=len(qas), desc="Generating answers", unit="qa")

        # Process in batches for checkpointing
        batch_size = max(
            1,
            self._checkpoint_manager._checkpoint_interval
            if self._checkpoint_manager
            else 10,
        )

        for i in range(0, len(qas), batch_size):
            batch = qas[i : i + batch_size]
            tasks = [_process_qa(qa) for qa in batch]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Answer generation failed: %s", result)
                    continue
                if result is not None:
                    answered_qas.append(result)
                progress.update(1)

            # Checkpoint after each batch
            self._maybe_checkpoint(answered_qas)

        progress.close()

        logger.info(
            "Completed answers for %d/%d questions",
            len(answered_qas) - (len(existing_qas) if existing_qas else 0),
            len(qas),
        )
        return answered_qas

    async def _generate_answer_with_refinement(
        self,
        qa: QA,
        dataset: Dataset,
        model_id: str,
        refinement_chunk_size: int,
        semaphore: asyncio.Semaphore,
        data_source_description: Optional[str] = None,
    ) -> Optional[QA]:
        def _extract_answer(response: str) -> str:
            answer_text = response.strip()
            return answer_text

        if not qa.doc_indices:
            logger.warning(
                "Skipping QA due to missing document indices: %s",
                qa.question,
            )
            return None

        chunks = list(self._batched(qa.doc_indices, refinement_chunk_size))
        if not chunks:
            logger.warning(
                "No document chunks available for QA '%s'; skipping",
                qa.question,
            )
            return None

        current_answer: Optional[str] = None

        for chunk_idx, chunk in enumerate(chunks):
            doc_contents: list[str] = []
            valid_indices: list[int] = []
            for doc_idx in chunk:
                doc_contents.append(str(dataset[doc_idx][CONTENT_COLUMN]))
                valid_indices.append(doc_idx)

            if not valid_indices:
                logger.warning(
                    "No valid documents found for QA '%s' in chunk %d; skipping chunk",
                    qa.question,
                    chunk_idx,
                )
                continue

            formatted_docs = self._format_documents(
                doc_contents,
                indices=valid_indices,
            )

            if chunk_idx == 0:
                prompt = ANSWER_GENERATION_PROMPT.format(
                    DOCUMENTS=formatted_docs,
                    QUESTION=qa.question,
                )
                prompt = format_prompt_with_description(prompt, data_source_description)

                initial_answer = await self._chat_completion_with_retry(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    semaphore=semaphore,
                    postprocess=_extract_answer,
                )
                if initial_answer is None:
                    logger.warning(
                        "Initial answer generation failed; skipping QA '%s'",
                        qa.question,
                    )
                    return None
                current_answer = initial_answer
            else:
                prompt = ANSWER_REFINMENT_PROMPT.format(
                    QUESTION=qa.question,
                    CURRENT_ANSWER=current_answer or "",
                    ADDITIONAL_DOCS=formatted_docs,
                )
                prompt = format_prompt_with_description(prompt, data_source_description)

                response = await self._chat_completion_with_retry(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    semaphore=semaphore,
                    postprocess=_extract_answer,
                )
                if response is None:
                    logger.warning(
                        "Refinement chunk %d skipped due to model failure; keeping current answer for QA '%s'",
                        chunk_idx,
                        qa.question,
                    )
                    continue

                current_answer = response
            logger.debug(f"Answer after refinement step {chunk_idx}: {current_answer}")

        if current_answer is None:
            logger.warning(
                "No answer produced for QA '%s'; skipping",
                qa.question,
            )
            return None

        qa.answer = current_answer
        return qa

    def _retrieve_supporting_documents_for_qas(
        self,
        qas: list[QA],
        dataset: Dataset,
        dataset_name: Optional[str],
        use_minimal_bm25: bool,
        max_answer_docs: int,
    ) -> None:
        if max_answer_docs <= 0:
            raise ValueError("max_answer_docs must be positive")

        logger.info(
            "Retrieving supporting documents for %d questions (max answer docs=%d)",
            len(qas),
            max_answer_docs,
        )

        retriever = (
            BM25Retriever(dataset=dataset, dataset_name=dataset_name)
            if use_minimal_bm25
            else HybridRetriever(
                dataset=dataset, dataset_name=dataset_name, rerank_threshold=1.0
            )
        )

        for qa in tqdm(qas, desc="Retrieving supporting documents", unit="qa"):
            retrieved_doc_ids = self._retrieve_supporting_docs(
                retriever=retriever,
                question=qa.question,
                top_k=max_answer_docs,
            )

            if not qa.doc_indices:
                logger.warning(
                    "QA missing original document index before retrieval; skipping additional docs: %s",
                    qa.question,
                )
                continue

            allowed_additional = max(0, max_answer_docs - len(qa.doc_indices))
            if allowed_additional == 0:
                continue

            original_doc_idx = qa.doc_indices[0]

            seen = set(qa.doc_indices)
            new_doc_ids: list[int] = []
            for doc_id in retrieved_doc_ids:
                if doc_id == original_doc_idx or doc_id in seen:
                    continue
                seen.add(doc_id)
                new_doc_ids.append(doc_id)
                if len(new_doc_ids) >= allowed_additional:
                    break

            if new_doc_ids:
                qa.doc_indices.extend(new_doc_ids)
        logger.info("Supporting documents attached to QA pairs")

    @staticmethod
    def _retrieve_supporting_docs(
        retriever: Any,
        question: str,
        top_k: int,
    ) -> list[int]:
        if top_k <= 0:
            return []

        retrieved: list[int] = []
        seen: set[int] = set()
        results = retriever.retrieve(
            question,
            top_k=top_k,
            rerank_batch_size=8,
        )
        for result in results:
            doc_id = result.doc_id
            if isinstance(doc_id, str):
                try:
                    doc_id = int(doc_id)
                except ValueError:
                    continue
            if isinstance(doc_id, int):
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                retrieved.append(doc_id)
                if len(retrieved) >= top_k:
                    break
        return retrieved

    @staticmethod
    def _batched(iterable: Iterable[int], size: int) -> Iterable[list[int]]:
        if size <= 0:
            raise ValueError("refinement_chunk_size must be positive")
        batch: list[int] = []
        for item in iterable:
            batch.append(item)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch


PIPELINE = AtomicQAPipeline
