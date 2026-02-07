import asyncio
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Sequence
from pathlib import Path

from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm.asyncio import tqdm

from ragent_core.data_sources import load_corpus
from ragent_core.pipelines.base import BasePipeline, CheckpointManager, PipelineMetadata
from ragent_core.pipelines.llms import DeepseekClient, GeminiClient
from ragent_core.pipelines.explorer_agent.explorer_agent_prompts import (
    BREADTH_AGENT_PROMPT,
    DEPTH_AGENT_PROMPT,
    SYNTHESIS_AGENT_PROMPT,
    extract_synthesized_qa,
    format_prompt_with_description,
)
from ragent_core.pipelines.prompts.atomic_prompts import (
    CONCEPT_EXTRACTOR_PROMPT,
    parse_concepts,
)
from ragent_core.pipelines.prompts.atomic_prompts import (
    format_prompt_with_description as format_concept_prompt_with_description,
)
from ragent_core.retrievers.bm25_retriever import BM25Retriever
from ragent_core.retrievers.hybrid_retriever import HybridRetriever
from ragent_core.types import QA, Concept

logger = logging.getLogger(__name__)

CONTENT_COLUMN = "text"
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_NUM_PAIRS = 100
DEFAULT_LLM_CLIENT = "gemini"
DEEPSEEK_CONCEPT_MODEL_ID = "deepseek-chat"
DEEPSEEK_BREADTH_MODEL_ID = "deepseek-reasoner"
DEEPSEEK_DEPTH_MODEL_ID = "deepseek-reasoner"
DEEPSEEK_SYNTHESIS_MODEL_ID = "deepseek-reasoner"
XIAOMI_CONCEPT_MODEL_ID = "mimo-v2-flash"
XIAOMI_BREADTH_MODEL_ID = "mimo-v2-flash"
XIAOMI_DEPTH_MODEL_ID = "mimo-v2-flash"
XIAOMI_SYNTHESIS_MODEL_ID = "mimo-v2-flash"
GEMINI_CONCEPT_MODEL_ID = "gemini-2.5-flash"
GEMINI_BREADTH_MODEL_ID = "gemini-3-flash-preview"
GEMINI_DEPTH_MODEL_ID = "gemini-3-flash-preview"
GEMINI_SYNTHESIS_MODEL_ID = "gemini-3-flash-preview"
DEFAULT_MAX_CONCURRENT_REQUESTS = 50
DEFAULT_ENABLE_TRAIN_EVAL_SPLIT = False
DEFAULT_TRAIN_SIZE = 0.9
DEFAULT_EVAL_SIZE = 0.1
DEFAULT_SEED = 42
DEFAULT_CHECKPOINT_INTERVAL = 10


@dataclass
class ExplorerAgentQAConfig:
    data_source: str
    num_pairs: int = DEFAULT_NUM_PAIRS
    seed: int = DEFAULT_SEED
    llm_client: str = DEFAULT_LLM_CLIENT
    concept_model_id: str = GEMINI_CONCEPT_MODEL_ID
    breadth_model_id: str = GEMINI_BREADTH_MODEL_ID
    depth_model_id: str = GEMINI_DEPTH_MODEL_ID
    synthesis_model_id: str = GEMINI_SYNTHESIS_MODEL_ID
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    enable_train_eval_split: bool = DEFAULT_ENABLE_TRAIN_EVAL_SPLIT
    train_size: float | int | None = DEFAULT_TRAIN_SIZE
    eval_size: float | int | None = DEFAULT_EVAL_SIZE
    use_bm25: bool = False
    # HybridRetriever configuration
    use_lite_retriever: bool = False
    # Checkpoint configuration
    checkpoint_dir: str | None = None
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    resume_from: str | None = None


@dataclass
class _StageContext:
    """Runtime context passed to stage handlers."""

    config: ExplorerAgentQAConfig
    dataset: Dataset
    data_source_description: str | None
    semaphore: asyncio.Semaphore
    rng: random.Random
    breadth_prompt: str
    depth_prompt: str
    synthesis_prompt: str


class ExplorerAgentCheckpointManager(CheckpointManager):
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
        pipeline_name: str = "pipeline",
        run_identifier: Optional[str] = None,
    ):
        super().__init__(
            checkpoint_dir, checkpoint_interval, pipeline_name, run_identifier
        )
        self._checkpoint_dir = (
            Path("data") / self._pipeline_name / "checkpoints" / self._run_identifier
        )
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        qas: Sequence[QA],
        concepts: Sequence[Concept],
        config: ExplorerAgentQAConfig,
        is_final: bool = False,
    ) -> Path:
        self._checkpoint_count += 1
        self._last_checkpoint_count = len(qas)
        if is_final:
            checkpoint_filename = "data.json"
        else:
            checkpoint_filename = (
                f"checkpoint_{self._checkpoint_count:04d}_{len(qas):06d}.json"
            )
        checkpoint_path = self._checkpoint_dir / checkpoint_filename
        config = asdict(config)
        config["num_generated_pairs"] = len(qas)
        checkpoint_data = {
            "config": config,
            "qas": [asdict(qa) for qa in qas],
            "concepts": [asdict(concept) for concept in concepts],
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        return checkpoint_path

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str | Path,
    ) -> tuple[Sequence[QA], Sequence[Concept]]:
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
        return [QA(**qa) for qa in checkpoint_data["qas"]], [
            Concept(**concept) for concept in checkpoint_data["concepts"]
        ]


class ExplorerAgentQAPipeline(BasePipeline):
    Config = ExplorerAgentQAConfig
    metadata = PipelineMetadata(
        name="qa/explorer_agent",
        description="Pipeline for generating QA pairs using explorer agent reasoning.",
    )

    def _init_llm_agent(self, config: ExplorerAgentQAConfig, retriever) -> None:
        client_name = config.llm_client.strip().lower()
        if client_name == "gemini":
            self._llm = GeminiClient(retriever)
        elif client_name == "deepseek":
            self._llm = DeepseekClient(retriever, provider="deepseek")
        elif client_name == "xiaomi":
            self._llm = DeepseekClient(retriever, provider="xiaomi")
        else:
            raise ValueError(f"Unsupported llm_client: {client_name}")

    def _load_dataset(
        self, data_source: str
    ) -> tuple[Dataset, Optional[str], Optional[str]]:
        logger.info("=" * 80)
        logger.info("Starting QA generation pipeline")
        logger.info("Data source: %s", data_source)
        logger.info("=" * 80)

        dataset, name, description = load_corpus(data_source)

        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected Dataset but got {type(dataset).__name__}")

        logger.info("Loaded dataset with %d documents", len(dataset))

        if CONTENT_COLUMN not in dataset.column_names:
            raise KeyError(
                f"Dataset '{data_source}' must include a '{CONTENT_COLUMN}' column after preprocessing."
            )

        return dataset, name, description

    async def _generate_concepts(
        self,
        dataset: Dataset,
        num_pairs: int,
        concept_model_id: str,
        rng: random.Random,
        semaphore: asyncio.Semaphore,
        data_source_description: str | None,
    ) -> list[Concept]:
        logger.info(
            "Generating at least %d concepts from dataset of %d documents",
            num_pairs,
            len(dataset),
        )

        if len(dataset) == 0:
            raise ValueError("Cannot generate concepts from an empty dataset")

        concepts: list[Concept] = []
        seen_names: set[str] = set()  # Track seen concept names during generation
        iteration = 0
        sample_size = min(DEFAULT_SAMPLE_SIZE, len(dataset))

        max_iterations = max(30, num_pairs // 5)
        max_no_progress = 5
        no_progress_count = 0

        while len(concepts) < num_pairs:
            if iteration >= max_iterations:
                logger.warning(
                    "Concept generation reached max iterations (%d) with %d/%d concepts.",
                    max_iterations,
                    len(concepts),
                    num_pairs,
                )
                break
            iteration += 1
            indices = self._sample_indices(rng, len(dataset), sample_size)
            subset = dataset.select(indices)

            prompt_documents = self._format_documents(
                subset[CONTENT_COLUMN], indices=indices
            )
            prompt = format_concept_prompt_with_description(
                CONCEPT_EXTRACTOR_PROMPT.format(DOCUMENTS=prompt_documents),
                data_source_description,
            )

            logger.debug(
                "Concept iteration %d: calling model with documents %s",
                iteration,
                indices,
            )

            try:
                response = await self._llm.chat_completion_with_retry(
                    model=concept_model_id,
                    content=prompt,
                    semaphore=semaphore,
                )
            except Exception as e:
                logger.error(f"Error generating concepts: {e}")
                continue
            if response is None:
                continue
            parsed = parse_concepts(
                response,
                dataset.info.dataset_name or "unknown",
            )

            # Deduplicate concepts during generation
            new_concepts_count = 0
            for concept in parsed:
                if concept.name not in seen_names:
                    seen_names.add(concept.name)
                    concepts.append(concept)
                    new_concepts_count += 1
                    if len(concepts) >= num_pairs:
                        break
            if new_concepts_count == 0:
                no_progress_count += 1
                if no_progress_count >= max_no_progress:
                    logger.warning(
                        "Concept generation produced no new concepts for %d consecutive iterations "
                        "(%d/%d concepts).",
                        no_progress_count,
                        len(concepts),
                        num_pairs,
                    )
                    break
            else:
                no_progress_count = 0

            logger.info(
                "[Iteration %d] +%d new concepts (total %d/%d) from documents %s",
                iteration,
                new_concepts_count,
                len(concepts),
                num_pairs,
                indices,
            )

        logger.info("Concept generation complete: %d unique concepts", len(concepts))
        return concepts

    async def _process_breadth_concept(
        self,
        concept: Concept,
        model_id: str,
        semaphore: asyncio.Semaphore,
        system_prompt: str,
    ) -> List[QA]:
        """Process a single concept for breadth generation."""
        qa_pairs: list[QA] = []
        breadth = concept.info["breadth"]
        content = f"Entity: {concept.name}, Max_Breadth: {breadth}"
        qa_pairs = await self._llm.run_agent(
            content,
            model_id,
            semaphore,
            system_prompt=system_prompt,
        )
        for qa in qa_pairs:
            qa.info = {
                "concept": concept.name,
                "breadth": breadth,
                "depth": concept.info["depth"],
            }
        return qa_pairs

    async def _process_depth_qa(
        self,
        qa: QA,
        model_id: str,
        semaphore: asyncio.Semaphore,
        system_prompt: str,
    ) -> Optional[QA]:
        """Process a single QA pair for depth generation."""
        depth = qa.info.get("depth")
        if not depth:
            logger.warning(
                f"QA pair for concept {qa.info.get('concept')} is missing depth information."
            )
            return qa
        if depth == 1:
            return qa

        qa_pairs = await self._llm.run_agent(
            f"<entity>{qa.info['concept']}</entity>\n<max_depth>{depth}</max_depth>\n<question>{qa.question}</question>\n<answer>{qa.answer}</answer>",
            model_id,
            semaphore,
            system_prompt=system_prompt,
        )

        if qa_pairs:
            # In case of multiple qa pairs, return the last one
            final_submission = qa_pairs[-1]
            final_submission.info = {**final_submission.info, **qa.info}
            return final_submission
        return None

    @staticmethod
    def _format_qa_pairs_for_prompt(qa_pairs: Sequence[QA]) -> str:
        lines: list[str] = ["<qa_pairs>"]
        for idx, qa in enumerate(qa_pairs, start=1):
            doc_ids = (
                " ".join(str(i) for i in qa.doc_ids)
                if qa.doc_ids
                else "No document IDs available"
            )
            lines.append("  <qa>")
            lines.append(f"    <id>{idx}</id>")
            lines.append(f"    <question>{qa.question.strip()}</question>")
            lines.append(f"    <answer>{qa.answer.strip()}</answer>")
            lines.append(f"    <doc_ids>{doc_ids}</doc_ids>")
            lines.append("  </qa>")
        lines.append("</qa_pairs>")
        return "\n".join(lines)

    async def _process_synthesis_group(
        self,
        concept: str,
        qa_pairs: List[QA],
        model_id: str,
        semaphore: asyncio.Semaphore,
        system_prompt: str,
    ) -> Optional[QA]:
        """Process a single group of QA pairs for synthesis."""
        # Filter out None values that may have been added from failed depth processing
        qa_pairs = [qa for qa in qa_pairs if qa is not None]

        if not qa_pairs:
            return None
        if len(qa_pairs) == 1:
            return qa_pairs[0]

        merged_doc_ids = set()
        for qa in qa_pairs:
            if qa and qa.doc_ids:
                merged_doc_ids.update(qa.doc_ids)
        merged_doc_ids = list(merged_doc_ids)

        qa_pairs_prompt = self._format_qa_pairs_for_prompt(qa_pairs)
        try:
            response = await self._llm.chat_completion_with_retry(
                model=model_id,
                content=f"Entity: {concept}\n\nQA_Pairs:\n{qa_pairs_prompt}",
                semaphore=semaphore,
                system_prompt=system_prompt,
                use_tools=False,
            )
        except Exception as e:
            logger.error(f"Error processing synthesis group: {e}")
            return None
        query, answer = extract_synthesized_qa(response)

        # Handle None values from extraction (only for query and answer now)
        if query is None or answer is None:
            logger.warning(f"Failed to extract synthesized QA for concept {concept}.")
            return None

        info = qa_pairs[0].info.copy()
        info["synthesis"] = True
        info["original_qa_pairs"] = [
            {
                "question": qa.question,
                "answer": qa.answer,
                "doc_ids": qa.doc_ids if qa.doc_ids else [],
            }
            for qa in qa_pairs
        ]
        return QA(
            question=query,
            answer=answer,
            doc_ids=merged_doc_ids,  # Use the merged doc_ids
            info=info,
        )

    async def generate(
        self,
        config: ExplorerAgentQAConfig,
    ) -> list[QA]:
        """Generate QA pairs using explorer agent reasoning."""
        # Handle resume from checkpoint
        qa_pairs: list[QA] = []
        concepts: list[Concept] = []
        if config.resume_from:
            qa_pairs, concepts = ExplorerAgentCheckpointManager.load_checkpoint(
                config.resume_from
            )

        # Setup dataset and retriever
        dataset, name, data_source_description = self._load_dataset(config.data_source)
        if config.use_bm25:
            retriever = BM25Retriever(dataset=dataset, dataset_name=name)
            logger.info("Using BM25Retriever for explorer agent pipeline")
        else:
            hybrid_kwargs = {}
            if config.use_lite_retriever:
                logger.info(
                    "Using lightweight HybridRetriever for explorer agent pipeline"
                )
                hybrid_kwargs["colbert_model_name"] = (
                    "mixedbread-ai/mxbai-edge-colbert-v0-32m"
                )
                hybrid_kwargs["reranker_model_name"] = (
                    "mixedbread-ai/mxbai-rerank-xsmall-v1"
                )
                hybrid_kwargs["rerank_threshold"] = 0.0
                logger.info(
                    "Using lightweight HybridRetriever for explorer agent pipeline"
                )
            else:
                logger.info(
                    "Using standard HybridRetriever for explorer agent pipeline"
                )
            retriever = HybridRetriever(
                dataset=dataset, dataset_name=name, **hybrid_kwargs
            )

        self._checkpoint_manager = ExplorerAgentCheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            checkpoint_interval=config.checkpoint_interval,
            pipeline_name=name or "data_source",
        )

        self._init_llm_agent(config, retriever)
        if self._llm is None:
            raise RuntimeError("LLM agent was not initialized")

        # Create runtime context for stage execution
        ctx = _StageContext(
            config=config,
            dataset=dataset,
            data_source_description=data_source_description,
            semaphore=asyncio.Semaphore(config.max_concurrent_requests),
            rng=random.Random(config.seed),
            breadth_prompt=format_prompt_with_description(
                BREADTH_AGENT_PROMPT, data_source_description
            ),
            depth_prompt=format_prompt_with_description(
                DEPTH_AGENT_PROMPT, data_source_description
            ),
            synthesis_prompt=format_prompt_with_description(
                SYNTHESIS_AGENT_PROMPT, data_source_description
            ),
        )

        if not concepts:
            concepts = await self._generate_concepts_with_breadth_depth(
                config.num_pairs, ctx
            )
            qa_pairs = await self._generate_qa_pairs_batch(
                concepts, ctx, existing_qa_pairs=[]
            )
        else:
            concepts_done = [qa.info["concept"] for qa in qa_pairs]
            concepts_left = [
                concept for concept in concepts if concept.name not in concepts_done
            ]
            logger.info(f"Generating {len(concepts_left)} concepts left")
            qa_pairs = await self._generate_qa_pairs_batch(
                concepts_left, ctx, existing_qa_pairs=qa_pairs
            )

        self._checkpoint_manager.save_checkpoint(
            qa_pairs, concepts, config, is_final=True
        )

        logger.info("=" * 80)
        logger.info("✓ Explorer agent QA generation complete!")
        logger.info("Final output: %d QA pairs", len(qa_pairs))
        logger.info("=" * 80)

        return qa_pairs

    async def _generate_qa_pair(self, concept: Concept, ctx: "_StageContext") -> QA:
        try:
            """Generate a QA pair for a concept."""
            logger.warning(
                f"Generating breadth for concept: {concept.name} with breadth: {concept.info['breadth']}"
            )
            qa_pairs = await self._process_breadth_concept(
                concept, ctx.config.breadth_model_id, ctx.semaphore, ctx.breadth_prompt
            )
            logger.warning(
                f"Genering depth for concept: {concept.name} with depth: {concept.info['depth']} for {len(qa_pairs)} qa pairs"
            )
            depth_qa_pairs = []
            for qa_pair in qa_pairs:
                qa_pairs = await self._process_depth_qa(
                    qa_pair, ctx.config.depth_model_id, ctx.semaphore, ctx.depth_prompt
                )
                depth_qa_pairs.append(qa_pairs)
            logger.warning(
                f"Generating synthesis for concept: {concept.name} with {len(depth_qa_pairs)} depth qa pairs"
            )
            qa_pair = await self._process_synthesis_group(
                concept.name,
                depth_qa_pairs,
                ctx.config.synthesis_model_id,
                ctx.semaphore,
                ctx.synthesis_prompt,
            )
            logger.warning(f"Generated QA pair for concept: {concept.name}")
            return qa_pair
        except Exception as e:
            logger.error(f"Error generating QA pair for concept: {concept.name}: {e}")
            return None

    async def _generate_qa_pairs_batch(
        self,
        concepts: list[Concept],
        ctx: "_StageContext",
        existing_qa_pairs: list[QA] = [],
    ) -> list[QA]:
        """Generate QA pairs in batches."""
        final_qa_pairs: list[QA] = existing_qa_pairs.copy()
        batch_size = max(
            1,
            self._checkpoint_manager._checkpoint_interval
            if self._checkpoint_manager
            else len(concepts),
        )
        num_batches = (len(concepts) + batch_size - 1) // batch_size
        logger.info(
            "Generating QA pairs in %d batch(es) (batch size: %d)",
            num_batches,
            batch_size,
        )
        for batch_start in range(0, len(concepts), batch_size):
            batch = concepts[batch_start : batch_start + batch_size]
            tasks = [self._generate_qa_pair(concept, ctx) for concept in batch]
            batch_results = await tqdm.gather(
                *tasks,
                desc=f"Generating QA pairs, batch {batch_start // batch_size + 1} of {num_batches}",
                total=len(batch),
            )
            for result in batch_results:
                if result is not None:
                    final_qa_pairs.append(result)
            self._checkpoint_manager.save_checkpoint(
                final_qa_pairs, concepts, ctx.config, is_final=False
            )
        return final_qa_pairs

    async def _generate_concepts_with_breadth_depth(
        self, target_pairs: int, ctx: "_StageContext"
    ) -> list[Concept]:
        """Generate concepts with breadth/depth config applied."""
        if target_pairs <= 0:
            return []
        concepts = await self._generate_concepts(
            dataset=ctx.dataset,
            num_pairs=target_pairs,
            concept_model_id=ctx.config.concept_model_id,
            rng=ctx.rng,
            semaphore=ctx.semaphore,
            data_source_description=ctx.data_source_description,
        )
        occurrence_counts = self._count_concept_occurrences(ctx.dataset, concepts)
        percentiles = self._occurrence_percentiles(list(occurrence_counts.values()))
        for concept in concepts:
            count = occurrence_counts.get(concept.name, 0)
            percentile = percentiles.get(count, 0.0)
            breadth, depth = self._percentile_to_breadth_depth(percentile)
            concept.info["breadth"] = breadth
            concept.info["depth"] = depth
            concept.info["occurrences"] = count
            concept.info["occurrence_percentile"] = percentile
            logger.info(
                f"Generated concept: {concept.name} with breadth: {breadth} and depth: {depth} with {count} occurrences and percentile: {percentile}"
            )
        return concepts

    @staticmethod
    def _count_concept_occurrences(
        dataset: Dataset, concepts: Sequence[Concept]
    ) -> dict[str, int]:
        """Count lowercase substring occurrences of each concept in the dataset."""
        counts = {concept.name: 0 for concept in concepts}
        if not concepts:
            return counts
        needles = [(concept.name, concept.name.strip().lower()) for concept in concepts]
        for text in dataset[CONTENT_COLUMN]:
            if not text:
                continue
            haystack = str(text).lower()
            for name, needle in needles:
                if not needle:
                    continue
                counts[name] += haystack.count(needle)
        return counts

    @staticmethod
    def _occurrence_percentiles(counts: list[int]) -> dict[int, float]:
        """Return percentile ranks (0-100) for each unique count value."""
        if not counts:
            return {}
        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        if n == 1:
            return {sorted_counts[0]: 100.0}
        percentiles: dict[int, float] = {}
        idx = 0
        while idx < n:
            value = sorted_counts[idx]
            start = idx
            while idx + 1 < n and sorted_counts[idx + 1] == value:
                idx += 1
            end = idx
            avg_rank = (start + end) / 2
            percentiles[value] = 100.0 * avg_rank / (n - 1)
            idx += 1
        return percentiles

    @staticmethod
    def _percentile_to_breadth_depth(percentile: float) -> tuple[int, int]:
        """Map a percentile into the fixed breadth/depth buckets."""
        bounded = max(0.0, min(100.0, percentile))
        if bounded < 10:
            return 1, 1
        if bounded < 20:
            return 1, 2
        if bounded < 30:
            return 1, 3
        if bounded < 40:
            return 2, 1
        if bounded < 50:
            return 2, 2
        if bounded < 70:
            return 3, 1
        if bounded < 80:
            return 2, 3
        if bounded < 90:
            return 3, 2
        return 3, 3

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
        """Split QA pairs into train/eval partitions, optionally stratified by breadth and depth."""
        qas_list = list(qas)
        labels = [
            f"{qa.info.get('breadth', 'unknown')}_{qa.info.get('depth', 'unknown')}"
            for qa in qas_list
        ]
        if stratify:
            label_counts = defaultdict(int)
            for label in labels:
                label_counts[label] += 1
            if any(count < 2 for count in label_counts.values()):
                logger.warning(
                    "Stratified split disabled due to labels with fewer than 2 samples."
                )
                stratify = False
        stratify_labels: Optional[Sequence[str]] = labels if stratify else None

        try:
            train_qas, eval_qas = train_test_split(
                qas_list,
                test_size=eval_size,
                train_size=train_size,
                random_state=seed,
                shuffle=shuffle,
                stratify=stratify_labels,
            )
        except ValueError as exc:
            if stratify_labels is not None:
                logger.warning(
                    "Stratified split failed; retrying without stratification: %s",
                    exc,
                )
                train_qas, eval_qas = train_test_split(
                    qas_list,
                    test_size=eval_size,
                    train_size=train_size,
                    random_state=seed,
                    shuffle=shuffle,
                    stratify=None,
                )
            else:
                raise

        return list(train_qas), list(eval_qas)

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
