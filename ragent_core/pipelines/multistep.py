from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Optional, Sequence

import backoff
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import umap
import hdbscan
from sklearn.model_selection import train_test_split

from ragent_core.config import OPENROUTER_API_KEY, OPENROUTER_URL
from ragent_core.pipelines.base import BasePipeline, PipelineMetadata
from ragent_core.data_sources import (
    get_data_source_loader,
    normalize_data_source_result,
)
from ragent_core.pipelines.prompts.multistep_prompts import (
    DEFAULT_RECIPE_KEY,
    build_answer_prompt,
    build_question_prompt,
    extract_answer_from_text,
    extract_question_from_text,
)
from ragent_core.types import QA

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"
DEFAULT_RANDOM_SEED = 43
DEFAULT_BREADTH_NEIGHBOR_RANGE = (2, 5)
DEFAULT_RANDOM_WALK_CHAIN_LENGTH_RANGE = (2, 5)
DEFAULT_UNRELATED_GROUP_RANGE = (2, 3)
DEFAULT_BREADTH_WEIGHT = 0.40
DEFAULT_RANDOM_WALK_WEIGHT = 0.0
DEFAULT_UNRELATED_WEIGHT = 0.30
DEFAULT_ATOMIC_WEIGHT = 0.30
DEFAULT_NUM_PAIRS = 40
DEFAULT_MMR_LAMBDA = 0.7
DEFAULT_NEAR_QUANTILE = 0.2
DEFAULT_FAR_QUANTILE = 0.8
DEFAULT_MEDIUM_LOW_QUANTILE = 0.4
DEFAULT_MEDIUM_HIGH_QUANTILE = 0.6
DEFAULT_QUESTION_MODEL_ID = "google/gemini-2.5-flash"
DEFAULT_ANSWER_MODEL_ID = "google/gemini-2.5-flash"
DEFAULT_MAX_CONCURRENT_REQUESTS = 64
DEFAULT_BREADTH_CLUSTER_MIN_SIZE = 3
DEFAULT_BREADTH_CLUSTER_MAX_SIZE = 10
DEFAULT_BREADTH_CLUSTER_MIN_PROBABILITY = 0.5
DEFAULT_BREADTH_UMAP_N_NEIGHBORS = 15
DEFAULT_BREADTH_UMAP_COMPONENTS = 10
DEFAULT_BREADTH_UMAP_MIN_DIST = 0.05
DEFAULT_BREADTH_HDBSCAN_MIN_SAMPLES = 1
DEFAULT_CHECKPOINT_INTERVAL = 10

load_dotenv()


@dataclass
class MultiStepQAConfig:
    """Configuration for composing multi-step QA pairs."""

    data_source: str
    source_data_path: str
    num_pairs: int = DEFAULT_NUM_PAIRS
    embedding_model: str = DEFAULT_EMBEDDING_MODEL_ID
    seed: int = DEFAULT_RANDOM_SEED
    breadth_neighbor_range: tuple[int, int] = DEFAULT_BREADTH_NEIGHBOR_RANGE
    random_walk_chain_length_range: tuple[int, int] = (
        DEFAULT_RANDOM_WALK_CHAIN_LENGTH_RANGE
    )
    unrelated_group_range: tuple[int, int] = DEFAULT_UNRELATED_GROUP_RANGE
    breadth_weight: float = DEFAULT_BREADTH_WEIGHT
    random_walk_weight: float = DEFAULT_RANDOM_WALK_WEIGHT
    unrelated_weight: float = DEFAULT_UNRELATED_WEIGHT
    atomic_weight: float = DEFAULT_ATOMIC_WEIGHT
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    breadth_cluster_min_size: int = DEFAULT_BREADTH_CLUSTER_MIN_SIZE
    breadth_cluster_max_size: int = DEFAULT_BREADTH_CLUSTER_MAX_SIZE
    breadth_cluster_min_probability: float = DEFAULT_BREADTH_CLUSTER_MIN_PROBABILITY
    breadth_umap_n_neighbors: int = DEFAULT_BREADTH_UMAP_N_NEIGHBORS
    breadth_umap_components: int = DEFAULT_BREADTH_UMAP_COMPONENTS
    breadth_umap_min_dist: float = DEFAULT_BREADTH_UMAP_MIN_DIST
    breadth_hdbscan_min_samples: int = DEFAULT_BREADTH_HDBSCAN_MIN_SAMPLES
    near_quantile: float = DEFAULT_NEAR_QUANTILE
    far_quantile: float = DEFAULT_FAR_QUANTILE
    medium_low_quantile: float = DEFAULT_MEDIUM_LOW_QUANTILE
    medium_high_quantile: float = DEFAULT_MEDIUM_HIGH_QUANTILE
    question_model_id: str = DEFAULT_QUESTION_MODEL_ID
    answer_model_id: str = DEFAULT_ANSWER_MODEL_ID
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    enable_train_eval_split: bool = True
    train_size: float | int | None = 0.9
    eval_size: float | int | None = 0.1
    # Checkpoint configuration
    checkpoint_dir: str | None = None
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    resume_from: str | None = None
    enable_checkpointing: bool = True


@dataclass
class MultiStepQA:
    """Aggregated QA bundle produced by a multi-step recipe."""

    questions: list[str]
    answers: list[str]
    doc_indices: list[int]
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class QARecord:
    """Atomic QA record loaded from the default pipeline output."""

    position: int
    source_id: int
    question: str
    answer: str
    doc_indices: list[int]
    info: dict[str, Any]


@dataclass
class EmbeddingMetrics:
    """Container for pre-computed embedding distances and neighbor data."""

    question_embeddings: np.ndarray
    answer_embeddings: np.ndarray
    question_similarities: np.ndarray
    question_distances: np.ndarray
    question_distances_raw: np.ndarray
    answer_distances: np.ndarray
    answer_distances_raw: np.ndarray
    neighbor_order: np.ndarray
    near_threshold: float
    far_threshold: float
    medium_low: float
    medium_high: float
    answer_far_threshold: float
    near_neighbors: list[np.ndarray]


class MultiStepQAPipeline(BasePipeline):
    """Compose multi-step QA pairs from atomic QAs using embedding-based recipes."""

    Config = MultiStepQAConfig
    metadata = PipelineMetadata(
        name="qa/multistep",
        description=(
            "Merge atomic QA pairs into multi-step reasoning prompts using clustering, graph random walks, "
            "multi-topic bundles, and direct atomic replay recipes."
        ),
    )

    def __init__(self, client: Optional[AsyncOpenAI] = None) -> None:
        super().__init__()
        self._client = client or AsyncOpenAI(
            base_url=OPENROUTER_URL,
            api_key=OPENROUTER_API_KEY,
        )

    def _load_data_source_description(
        self, data_source: Optional[str]
    ) -> Optional[str]:
        """Load the data source description from the registered loader, if available."""
        if not data_source:
            return None

        try:
            loader = get_data_source_loader(data_source)
        except (ModuleNotFoundError, AttributeError):
            return None

        try:
            result = loader()
            spec = normalize_data_source_result(result)
            return spec.description
        except Exception as e:
            logger.warning(
                "Failed to load data source description for %s: %s",
                data_source,
                e,
            )
            return None

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter
    )
    async def _chat_completion_with_retry(
        self,
        model: str,
        messages: list[dict[str, Any]],
        semaphore: asyncio.Semaphore,
        postprocess: Optional[Callable[[Any], Any | Awaitable[Any]]] = None,
        temperature: float = 0.8,
    ) -> Optional[Any]:
        """Invoke the chat completion API with optional post-processing and retries."""
        async with semaphore:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                # temperature=temperature,
            )
            if postprocess is None:
                return response

            processed = postprocess(response)
            if inspect.isawaitable(processed):
                processed = await processed
            return processed

    async def generate(self, config: MultiStepQAConfig) -> list[QA]:
        """Orchestrate multi-step QA synthesis according to the configured recipes."""
        if config.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")

        # Initialize checkpointing
        resumed_qas: list[QA] = []
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
            # If we already have enough QA pairs, return early
            if len(resumed_qas) >= config.num_pairs:
                logger.info(
                    "Already have %d QA pairs from checkpoint, no more needed",
                    len(resumed_qas),
                )
                return resumed_qas[: config.num_pairs]

        data_source_description = self._load_data_source_description(config.data_source)

        records = self._load_source(Path(config.source_data_path))
        if not records:
            logger.warning(
                "No QA records were loaded from '%s'; returning empty result set",
                config.source_data_path,
            )
            return resumed_qas

        metrics = await asyncio.to_thread(self._build_metrics, records, config)
        rng = np.random.default_rng(config.seed)

        # Calculate remaining pairs needed
        remaining_pairs = max(0, config.num_pairs - len(resumed_qas))

        weighted_recipes = [
            ("breadth", config.breadth_weight),
            ("random_walk", config.random_walk_weight),
            ("unrelated", config.unrelated_weight),
            ("atomic", config.atomic_weight),
        ]

        recipe_targets = self._allocate_recipe_targets(
            remaining_pairs,
            weighted_recipes,
        )

        bundles: list[MultiStepQA] = []
        recipe_generators = [
            ("breadth", self._generate_breadth_groups),
            ("random_walk", self._generate_random_walk_chains),
            ("unrelated", self._generate_unrelated_queries),
            ("atomic", self._generate_atomic_samples),
        ]

        for recipe_name, generator in recipe_generators:
            remaining = max(0, remaining_pairs - len(bundles))
            if remaining == 0:
                break
            target = min(recipe_targets[recipe_name], remaining)
            if target <= 0:
                continue
            generated = generator(records, metrics, config, rng, target)
            bundles.extend(generated)

        initial_bundle_count = len(bundles)
        if bundles:
            bundles = self._deduplicate_bundles(bundles)
        if len(bundles) < initial_bundle_count:
            logger.info(
                "Removed %d duplicate bundles prior to final composition",
                initial_bundle_count - len(bundles),
            )

        if len(bundles) > remaining_pairs:
            bundles = bundles[:remaining_pairs]

        if len(bundles) < remaining_pairs:
            logger.info(
                "Requested %d multi-step groups but generated %d due to candidate limits",
                remaining_pairs,
                len(bundles),
            )

        bundles = bundles[:remaining_pairs]

        final_qas = await self._compose_final_pairs(
            bundles,
            config,
            existing_qas=resumed_qas,
            data_source_description=data_source_description,
        )

        # Save final checkpoint
        self._save_final_checkpoint(final_qas)

        logger.info(
            "Multi-step QA pipeline produced %d final QA pairs from %d grouped bundles",
            len(final_qas) - len(resumed_qas),
            len(bundles),
        )
        return final_qas

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
        """Split QA pairs into train/eval partitions, optionally stratified by recipe."""
        qas_list = list(qas)
        labels = [str(qa.info.get("recipe", DEFAULT_RECIPE_KEY)) for qa in qas_list]
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

    @staticmethod
    def _load_source(path: Path) -> list[QARecord]:
        """Load and normalize atomic QA records from the provided JSON artifact."""
        if not path.exists():
            raise FileNotFoundError(f"Input data file '{path}' does not exist")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, list):
            raise ValueError(
                f"Expected list payload in '{path}', received {type(payload)!r}"
            )

        records: list[QARecord] = []
        for position, item in enumerate(payload):
            if not isinstance(item, dict):
                logger.debug(
                    "Skipping non-dict QA entry at position %d: %r", position, item
                )
                continue

            question = item.get("question", "").strip()
            answer = str(item.get("answer", "")).strip()

            if not question or not answer:
                logger.debug(
                    "Skipping QA entry with empty question/answer at position %d",
                    position,
                )
                continue

            doc_indices = item.get("doc_indices", [])
            info = item.get("info") if isinstance(item.get("info"), dict) else {}

            source_id = item.get("id", position + 1)
            try:
                source_id = int(source_id)
            except (TypeError, ValueError):
                source_id = position + 1

            records.append(
                QARecord(
                    position=position,
                    source_id=source_id,
                    question=question,
                    answer=answer,
                    doc_indices=doc_indices,
                    info=info,
                )
            )

        return records

    def _build_metrics(
        self,
        records: Sequence[QARecord],
        config: MultiStepQAConfig,
    ) -> EmbeddingMetrics:
        """Compute embeddings, distance matrices, and neighbor orderings."""
        model = SentenceTransformer(
            config.embedding_model,
        )

        question_texts = [record.question for record in records]
        answer_texts = [record.answer for record in records]

        question_embeddings = model.encode(
            question_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt_name="query",
        )
        answer_embeddings = model.encode(
            answer_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        question_tensor = torch.from_numpy(question_embeddings)
        logger.info(f"Question embeddings shape: {question_tensor.shape}")
        question_similarities = cos_sim(question_tensor, question_tensor).cpu().numpy()
        question_distances_raw = 1.0 - question_similarities
        np.fill_diagonal(question_distances_raw, 0.0)

        question_distances = question_distances_raw.copy()
        np.fill_diagonal(question_distances, np.inf)
        neighbor_order = np.argsort(question_distances, axis=1)

        answer_tensor = torch.from_numpy(answer_embeddings)
        logger.info(f"Answer embeddings shape: {answer_tensor.shape}")
        answer_similarities = cos_sim(answer_tensor, answer_tensor).cpu().numpy()
        answer_distances_raw = 1.0 - answer_similarities
        np.fill_diagonal(answer_distances_raw, 0.0)

        answer_distances = answer_distances_raw.copy()
        np.fill_diagonal(answer_distances, np.inf)

        thresholds = self._compute_thresholds(
            question_distances_raw,
            answer_distances_raw,
            config,
        )

        if np.isfinite(thresholds["near"]):
            near_neighbors = [
                np.array(
                    [
                        neighbor
                        for neighbor in neighbor_order[idx]
                        if neighbor != idx
                        and np.isfinite(question_distances_raw[idx, neighbor])
                        and question_distances_raw[idx, neighbor] <= thresholds["near"]
                    ],
                    dtype=int,
                )
                for idx in range(len(records))
            ]
        else:
            near_neighbors = [np.array([], dtype=int) for _ in range(len(records))]

        return EmbeddingMetrics(
            question_embeddings=question_embeddings,
            answer_embeddings=answer_embeddings,
            question_similarities=question_similarities,
            question_distances=question_distances,
            question_distances_raw=question_distances_raw,
            answer_distances=answer_distances,
            answer_distances_raw=answer_distances_raw,
            neighbor_order=neighbor_order,
            near_threshold=thresholds["near"],
            far_threshold=thresholds["far"],
            medium_low=thresholds["medium_low"],
            medium_high=thresholds["medium_high"],
            answer_far_threshold=thresholds["answer_far"],
            near_neighbors=near_neighbors,
        )

    @staticmethod
    def _compute_thresholds(
        question_distances: np.ndarray,
        answer_distances: np.ndarray,
        config: MultiStepQAConfig,
    ) -> dict[str, float]:
        """Derive near/medium/far band cutoffs using dataset-level quantiles."""
        n = question_distances.shape[0]
        if n < 2:
            return {
                "near": float("inf"),
                "far": float("inf"),
                "medium_low": float("inf"),
                "medium_high": float("inf"),
                "answer_far": float("inf"),
            }

        tri_upper = np.triu_indices(n, k=1)
        question_pairs = question_distances[tri_upper]
        answer_pairs = answer_distances[tri_upper]

        def _quantile(values: np.ndarray, q: float) -> float:
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                return float("inf")
            return float(np.quantile(finite_values, q, method="nearest"))

        return {
            "near": _quantile(question_pairs, config.near_quantile),
            "far": _quantile(question_pairs, config.far_quantile),
            "medium_low": _quantile(question_pairs, config.medium_low_quantile),
            "medium_high": _quantile(question_pairs, config.medium_high_quantile),
            "answer_far": _quantile(answer_pairs, config.far_quantile),
        }

    @staticmethod
    def _sample_range(range_: tuple[int, int], rng: np.random.Generator) -> int:
        """Sample an integer within an inclusive range."""
        low, high = range_
        if high < low:
            low, high = high, low
        if low == high:
            return low
        return int(rng.integers(low, high + 1))

    @staticmethod
    def _deduplicate_bundles(
        bundles: Sequence[MultiStepQA],
    ) -> list[MultiStepQA]:
        """Drop duplicate bundles based on recipe, sources, questions, and answers."""
        seen: set[tuple[Any, ...]] = set()
        deduped: list[MultiStepQA] = []

        for bundle in bundles:
            recipe = str(bundle.info.get("recipe", "")).casefold()
            source_ids = tuple(bundle.info.get("source_ids", ()))
            normalized_questions = tuple(q.strip().casefold() for q in bundle.questions)
            normalized_answers = tuple(a.strip().casefold() for a in bundle.answers)
            key = (recipe, source_ids, normalized_questions, normalized_answers)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(bundle)

        return deduped

    @staticmethod
    def _allocate_recipe_targets(
        num_pairs: int,
        weighted_recipes: Sequence[tuple[str, float]],
    ) -> dict[str, int]:
        """Allocate a target count per recipe using weighted rounding."""
        if num_pairs <= 0:
            return {name: 0 for name, _ in weighted_recipes}

        normalized_weights = [max(weight, 0.0) for _, weight in weighted_recipes]
        total_weight = sum(normalized_weights)
        if total_weight <= 0:
            normalized_weights = [1.0 for _ in weighted_recipes]
            total_weight = float(len(weighted_recipes))

        raw_allocations: list[tuple[str, float]] = []
        allocations: dict[str, int] = {}

        for idx, (name, _) in enumerate(weighted_recipes):
            weight = normalized_weights[idx]
            raw_value = (num_pairs * weight) / total_weight
            raw_allocations.append((name, raw_value))
            allocations[name] = int(raw_value)

        assigned = sum(allocations.values())
        remainder = num_pairs - assigned

        if remainder > 0:
            remainders = sorted(
                (
                    (name, raw_value - allocations[name], idx)
                    for idx, (name, raw_value) in enumerate(raw_allocations)
                ),
                key=lambda item: (-item[1], item[2]),
            )
            for name, _, _ in remainders:
                if remainder <= 0:
                    break
                allocations[name] += 1
                remainder -= 1
        elif remainder < 0:
            # Float rounding should not result in over-assignment, but guard just in case.
            reductions = sorted(
                (
                    (name, allocations[name] - raw_value, idx)
                    for idx, (name, raw_value) in enumerate(raw_allocations)
                    if allocations[name] > 0
                ),
                key=lambda item: (-item[1], item[2]),
            )
            for name, _, _ in reductions:
                if remainder >= 0:
                    break
                allocations[name] -= 1
                remainder += 1

        breadth_allocation = allocations.get("breadth", 0)
        if breadth_allocation > 0 and breadth_allocation % 2 != 0:
            for idx, (name, _) in enumerate(weighted_recipes):
                if name == "breadth":
                    continue
                allocations[name] = allocations.get(name, 0) + 1
                allocations["breadth"] -= 1
                logger.debug(
                    "Adjusted breadth allocation to %d to ensure paired prompts; shifted slot to '%s'",
                    allocations["breadth"],
                    name,
                )
                break
            else:
                logger.debug(
                    "Breadth allocation %d could not be paired; leaving unchanged",
                    breadth_allocation,
                )

        return allocations

    def _generate_breadth_groups(
        self,
        records: Sequence[QARecord],
        metrics: EmbeddingMetrics,
        config: MultiStepQAConfig,
        rng: np.random.Generator,
        target_count: int,
    ) -> list[MultiStepQA]:
        """Group topically related QAs to encourage breadth-first synthesis."""
        if target_count <= 0 or not records:
            return []

        effective_target = target_count
        if effective_target % 2 != 0:
            logger.debug(
                "Reducing breadth target from %d to %d to generate paired prompts",
                effective_target,
                effective_target - 1,
            )
            effective_target -= 1
        if effective_target <= 0:
            return []

        used_indices: set[int] = set()

        outputs, _ = self._generate_breadth_groups_hdbscan(
            records,
            metrics,
            config,
            rng,
            effective_target,
            used_indices,
        )

        logger.info(
            "Breadth-first recipe produced %d QA groups via UMAP+HDBSCAN clustering",
            len(outputs),
        )
        return outputs[:effective_target]

    def _generate_breadth_groups_hdbscan(
        self,
        records: Sequence[QARecord],
        metrics: EmbeddingMetrics,
        config: MultiStepQAConfig,
        rng: np.random.Generator,
        target_count: int,
        used_indices: set[int],
    ) -> tuple[list[MultiStepQA], list[int]]:
        """Generate breadth groups via UMAP dimensionality reduction and HDBSCAN clustering."""
        outputs: list[MultiStepQA] = []
        noise_candidates: list[int] = []

        if metrics.question_embeddings.size == 0:
            return outputs, noise_candidates

        min_size = max(2, config.breadth_cluster_min_size)
        max_size = max(min_size, config.breadth_cluster_max_size)

        reducer = umap.UMAP(
            n_neighbors=max(config.breadth_umap_n_neighbors, min_size + 2),
            n_components=max(2, config.breadth_umap_components),
            min_dist=max(0.0, config.breadth_umap_min_dist),
            metric="cosine",
            random_state=config.seed,
        )
        embedding = reducer.fit_transform(metrics.question_embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            min_samples=max(1, config.breadth_hdbscan_min_samples),
            metric="euclidean",
            cluster_selection_method="leaf",
            allow_single_cluster=False,
            prediction_data=True,
        )
        labels = clusterer.fit_predict(embedding)
        logger.info(f"Number of clusters: {len(np.unique(labels))}")
        probabilities = getattr(clusterer, "probabilities_", None)
        if probabilities is None:
            probabilities = np.ones(len(records), dtype=float)

        unchunked_groups: list[tuple[int, list[int]]] = []
        chunked_groups: list[tuple[int, list[int]]] = []

        for label in np.unique(labels):
            members = np.where(labels == label)[0]
            if label < 0:
                # HDBSCAN assigns a noise label of -1 (and other negative values) for outliers.
                noise_candidates.extend(int(idx) for idx in members)
                continue

            filtered_members = [
                int(idx)
                for idx in members
                if probabilities[int(idx)] >= config.breadth_cluster_min_probability
            ]

            if len(filtered_members) < min_size:
                noise_candidates.extend(filtered_members)
                continue

            rng.shuffle(filtered_members)
            if len(filtered_members) <= max_size:
                unchunked_groups.append((int(label), filtered_members))
            else:
                # Break oversized clusters into small chunks to respect the desired max group size.
                start = 0
                while len(filtered_members) - start >= min_size:
                    chunk = filtered_members[start : start + max_size]
                    chunked_groups.append((int(label), chunk))
                    start += max_size
                remainder = filtered_members[start:]
                if remainder:
                    noise_candidates.extend(remainder)

        if not unchunked_groups and not chunked_groups:
            return outputs, noise_candidates

        rng.shuffle(unchunked_groups)
        rng.shuffle(chunked_groups)

        def _consume_groups(
            groups: Sequence[tuple[int, list[int]]], chunked: bool
        ) -> None:
            nonlocal outputs
            for label, members in groups:
                if len(outputs) >= target_count:
                    break

                members = [idx for idx in members if idx not in used_indices]
                if len(members) < min_size:
                    noise_candidates.extend(members)
                    continue

                group_indices = self._select_cluster_group(
                    members,
                    metrics,
                    config,
                )
                if len(group_indices) < min_size:
                    noise_candidates.extend(members)
                    continue
                if any(idx in used_indices for idx in group_indices):
                    continue

                probs = [float(probabilities[idx]) for idx in group_indices]
                capacity = target_count - len(outputs)
                if capacity < 2:
                    break

                base_info = {
                    "breadth_strategy": "cluster",
                    "cluster_label": label,
                    "cluster_candidate_size": len(members),
                    "cluster_chunked": chunked,
                    "cluster_probability_min": min(probs),
                    "cluster_probability_max": max(probs),
                    "umap_n_neighbors": config.breadth_umap_n_neighbors,
                    "umap_n_components": config.breadth_umap_components,
                }
                used_indices.update(group_indices)
                variant_groups = [
                    self._compose_group(
                        group_indices,
                        records,
                        recipe="breadth_multihop_logic_search",
                        extra_info=dict(
                            base_info, breadth_prompt_variant="multihop_logic_search"
                        ),
                    ),
                    self._compose_group(
                        group_indices,
                        records,
                        recipe="breadth_thematic_synthesis",
                        extra_info=dict(
                            base_info, breadth_prompt_variant="thematic_synthesis"
                        ),
                    ),
                ]
                outputs.extend(variant_groups)

        _consume_groups(unchunked_groups, chunked=False)
        if len(outputs) < target_count:
            _consume_groups(chunked_groups, chunked=True)

        return outputs, noise_candidates

    def _select_cluster_group(
        self,
        members: Sequence[int],
        metrics: EmbeddingMetrics,
        config: MultiStepQAConfig,
    ) -> list[int]:
        """Select a representative subset from a cluster."""
        unique_members = list(dict.fromkeys(int(idx) for idx in members))
        if len(unique_members) < max(2, config.breadth_cluster_min_size):
            return []

        distances = metrics.question_distances_raw
        similarities = metrics.question_similarities

        def _mean_distance(idx: int) -> float:
            others = [member for member in unique_members if member != idx]
            if not others:
                return float("inf")
            return float(np.mean(distances[idx, others]))

        anchor = min(unique_members, key=_mean_distance)
        candidate_order = sorted(
            (member for member in unique_members if member != anchor),
            key=lambda idx: similarities[anchor, idx],
            reverse=True,
        )
        neighbor_limit = len(unique_members) - 1
        selection = self._mmr_select_neighbors(
            anchor,
            candidate_order,
            similarities,
            neighbor_limit,
            config.mmr_lambda,
        )
        group_indices = [anchor, *selection]

        max_size = max(config.breadth_cluster_min_size, config.breadth_cluster_max_size)
        if len(group_indices) > max_size:
            group_indices = group_indices[:max_size]
        return group_indices

    def _generate_random_walk_chains(
        self,
        records: Sequence[QARecord],
        metrics: EmbeddingMetrics,
        config: MultiStepQAConfig,
        rng: np.random.Generator,
        target_count: int,
    ) -> list[MultiStepQA]:
        """Form chains of random walks to create sub-graphs. I haven't gotten good qa pairs from this formula for now so it's deactivated for now.
        The groups it picks are too similar to just unrelated/multitopic queries"""
        outputs: list[MultiStepQA] = []
        if target_count <= 0 or not records:
            return outputs

        candidate_nodes = [
            idx
            for idx, neighbors in enumerate(metrics.near_neighbors)
            if neighbors.size > 0
        ]
        if not candidate_nodes:
            logger.info(
                "Random walk first recipe skipped because no near-neighbor graph could be formed"
            )
            return outputs

        max_attempts = max(len(candidate_nodes) * target_count * 2, target_count * 5)
        attempts = 0

        while len(outputs) < target_count and attempts < max_attempts:
            chain_length = self._sample_range(
                config.random_walk_chain_length_range, rng
            )
            anchor = int(rng.choice(candidate_nodes))
            chain = [anchor]
            visited = {anchor}
            current = anchor

            while len(chain) < chain_length:
                neighbors = metrics.near_neighbors[current]
                if neighbors.size == 0:
                    break

                unvisited = [int(idx) for idx in neighbors if idx not in visited]
                candidate_pool = unvisited or [
                    int(idx) for idx in neighbors if idx != current
                ]
                if not candidate_pool:
                    break

                candidate_array = np.array(candidate_pool, dtype=int)
                distances = metrics.question_distances_raw[current, candidate_array]
                finite_mask = np.isfinite(distances)
                if not np.all(finite_mask):
                    candidate_array = candidate_array[finite_mask]
                if candidate_array.size == 0:
                    break

                next_idx = int(rng.choice(candidate_array))
                chain.append(next_idx)
                visited.add(next_idx)
                current = next_idx

            if len(chain) == chain_length:
                hop_distances = [
                    float(metrics.question_distances_raw[chain[i], chain[i + 1]])
                    for i in range(len(chain) - 1)
                ]
                outputs.append(
                    self._compose_group(
                        chain,
                        records,
                        recipe="random_walk_chain",
                        extra_info={
                            "chain_length": chain_length,
                            "hop_distances": hop_distances,
                            "start_source_id": records[anchor].source_id,
                        },
                    )
                )
            attempts += 1

        logger.info(
            "Random walk recipe produced %d QA chains after %d attempts",
            len(outputs),
            attempts,
        )
        return outputs

    def _generate_unrelated_queries(
        self,
        records: Sequence[QARecord],
        metrics: EmbeddingMetrics,
        config: MultiStepQAConfig,
        rng: np.random.Generator,
        target_count: int,
    ) -> list[MultiStepQA]:
        """Bundle two or three distant questions to mimic multi-topic user asks."""
        outputs: list[MultiStepQA] = []
        attempts = 0
        max_attempts = len(records) * 5
        require_far = np.isfinite(metrics.far_threshold)

        while len(outputs) < target_count and attempts < max_attempts:
            requested_size = max(
                2, self._sample_range(config.unrelated_group_range, rng)
            )
            anchor = int(rng.integers(0, len(records)))
            group_indices = [anchor]

            for candidate in rng.permutation(len(records)):
                candidate = int(candidate)
                if candidate == anchor or candidate in group_indices:
                    continue
                if require_far and not all(
                    metrics.question_distances_raw[existing, candidate]
                    >= metrics.far_threshold
                    for existing in group_indices
                ):
                    continue
                group_indices.append(candidate)
                if len(group_indices) >= requested_size:
                    break

            if len(group_indices) < requested_size:
                attempts += 1
                continue

            pairwise_distances = [
                {
                    "source_a": records[group_indices[i]].source_id,
                    "source_b": records[group_indices[j]].source_id,
                    "distance": float(
                        metrics.question_distances_raw[
                            group_indices[i], group_indices[j]
                        ]
                    ),
                }
                for i in range(len(group_indices))
                for j in range(i + 1, len(group_indices))
            ]

            outputs.append(
                self._compose_group(
                    group_indices,
                    records,
                    recipe="multi_topic_queries",
                    extra_info={
                        "anchor_id": records[anchor].source_id,
                        "requested_size": requested_size,
                        "pairwise_distances": pairwise_distances,
                    },
                )
            )
            attempts = 0

        logger.info("Multi-topic bundle recipe produced %d QA groups", len(outputs))
        return outputs

    def _generate_atomic_samples(
        self,
        records: Sequence[QARecord],
        metrics: EmbeddingMetrics,
        config: MultiStepQAConfig,
        rng: np.random.Generator,
        target_count: int,
    ) -> list[MultiStepQA]:
        """Select single QA pairs directly from the source data for easy difficulty."""
        if target_count <= 0 or not records:
            return []

        outputs: list[MultiStepQA] = []
        candidates = list(range(len(records)))
        rng.shuffle(candidates)

        for idx in candidates:
            outputs.append(
                self._compose_group(
                    [idx],
                    records,
                    recipe="atomic_replay",
                    extra_info={
                        "source_id": records[idx].source_id,
                        "position": records[idx].position,
                    },
                )
            )
            if len(outputs) >= target_count:
                break

        logger.info("Atomic replay recipe produced %d QA pairs", len(outputs))
        return outputs

    async def _compose_final_pairs(
        self,
        bundles: Sequence[MultiStepQA],
        config: MultiStepQAConfig,
        existing_qas: list[QA] | None = None,
        data_source_description: Optional[str] = None,
    ) -> list[QA]:
        """Generate final user-facing questions and answers for each bundle via LLM."""
        all_qas: list[QA] = list(existing_qas) if existing_qas else []

        if not bundles:
            return all_qas

        semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        # Separate atomic and LLM bundles
        atomic_bundles: list[tuple[int, MultiStepQA]] = []
        llm_bundles: list[tuple[int, MultiStepQA]] = []

        for idx, bundle in enumerate(bundles):
            if str(bundle.info.get("recipe", "")).casefold() == "atomic_replay":
                atomic_bundles.append((idx, bundle))
            else:
                llm_bundles.append((idx, bundle))

        # Process atomic bundles directly (no LLM needed)
        for idx, bundle in atomic_bundles:
            if bundle.questions and bundle.answers:
                info = dict(bundle.info)
                info.update(
                    {
                        "sub_questions": bundle.questions,
                        "sub_answers": bundle.answers,
                    }
                )
                all_qas.append(
                    QA(
                        question=bundle.questions[0],
                        answer=bundle.answers[0],
                        doc_indices=bundle.doc_indices,
                        info=info,
                    )
                )
                self._maybe_checkpoint(all_qas)

        # Process LLM bundles with checkpointing
        if llm_bundles:
            batch_size = max(
                1,
                self._checkpoint_manager._checkpoint_interval
                if self._checkpoint_manager
                else 10,
            )

            for batch_start in range(0, len(llm_bundles), batch_size):
                batch = llm_bundles[batch_start : batch_start + batch_size]
                batch_bundles = [b for _, b in batch]

                # Generate questions for batch
                questions = await self._generate_final_questions(
                    batch_bundles,
                    model_id=config.question_model_id,
                    semaphore=semaphore,
                    seed=config.seed,
                    data_source_description=data_source_description,
                )

                # Generate answers for batch
                answers = await self._generate_final_answers(
                    batch_bundles,
                    questions,
                    model_id=config.answer_model_id,
                    semaphore=semaphore,
                    seed=config.seed,
                    data_source_description=data_source_description,
                )

                # Create QA objects
                for (_, bundle), question, answer in zip(batch, questions, answers):
                    if not question or not answer:
                        continue
                    info = dict(bundle.info)
                    info.update(
                        {
                            "sub_questions": bundle.questions,
                            "sub_answers": bundle.answers,
                        }
                    )
                    all_qas.append(
                        QA(
                            question=question,
                            answer=answer,
                            doc_indices=bundle.doc_indices,
                            info=info,
                        )
                    )

                # Checkpoint after each batch
                self._maybe_checkpoint(all_qas)

        return all_qas

    async def _generate_final_questions(
        self,
        bundles: Sequence[MultiStepQA],
        model_id: str,
        semaphore: asyncio.Semaphore,
        seed: int,
        data_source_description: Optional[str] = None,
    ) -> list[Optional[str]]:
        """Use the LLM to synthesize a single question per bundle."""
        tasks: list[Awaitable[Optional[str]]] = []
        rng = np.random.default_rng(seed)

        for index, bundle in enumerate(bundles):
            recipe = bundle.info.get("recipe", DEFAULT_RECIPE_KEY)
            prompt = build_question_prompt(
                recipe,
                bundle.questions,
                bundle.answers,
                rng=rng,
                data_source_description=data_source_description,
            )

            def _parse(response: Any, idx: int = index) -> Optional[str]:
                content = response.choices[0].message.content
                return extract_question_from_text(content)

            tasks.append(
                self._chat_completion_with_retry(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    semaphore=semaphore,
                    postprocess=_parse,
                )
            )

        responses = await tqdm_asyncio.gather(
            *tasks, desc="Generating multistep questions", unit="question"
        )
        return [resp if isinstance(resp, str) else None for resp in responses]

    async def _generate_final_answers(
        self,
        bundles: Sequence[MultiStepQA],
        questions: Sequence[Optional[str]],
        model_id: str,
        semaphore: asyncio.Semaphore,
        seed: int,
        data_source_description: Optional[str] = None,
    ) -> list[Optional[str]]:
        """Use the LLM to synthesize a final answer per bundle."""
        results: list[Optional[str]] = [None] * len(bundles)
        scheduled: list[tuple[int, Awaitable[Optional[str]]]] = []
        rng = np.random.default_rng(seed)

        for index, (bundle, question) in enumerate(zip(bundles, questions)):
            if not question:
                continue

            recipe = bundle.info.get("recipe", DEFAULT_RECIPE_KEY)
            prompt = build_answer_prompt(
                recipe,
                question,
                bundle.questions,
                bundle.answers,
                rng=rng,
                data_source_description=data_source_description,
            )

            def _parse(response: Any, idx: int = index) -> Optional[str]:
                content = response.choices[0].message.content
                return extract_answer_from_text(content)

            awaitable = self._chat_completion_with_retry(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                semaphore=semaphore,
                postprocess=_parse,
            )
            scheduled.append((index, awaitable))

        if scheduled:
            responses = await tqdm_asyncio.gather(
                *(awaitable for _, awaitable in scheduled),
                desc="Generating multistep answers",
                unit="answer",
            )
            for (idx, _), value in zip(scheduled, responses):
                results[idx] = value if isinstance(value, str) else None

        return results

    @staticmethod
    def _mmr_select_neighbors(
        anchor: int,
        candidates: Sequence[int],
        similarities: np.ndarray,
        limit: int,
        lambda_param: float,
    ) -> list[int]:
        """Select neighbors via Maximal Marginal Relevance to avoid redundancy."""
        if limit <= 0:
            return []

        selected: list[int] = []
        candidate_list = list(candidates)
        if not candidate_list:
            return selected

        selected.append(candidate_list[0])

        while len(selected) < min(limit, len(candidate_list)):
            best_candidate = None
            best_score = -np.inf

            for candidate in candidate_list:
                if candidate in selected:
                    continue
                relevance = similarities[anchor, candidate]
                redundancy = (
                    max(similarities[candidate, idx] for idx in selected)
                    if selected
                    else 0.0
                )
                score = lambda_param * relevance - (1 - lambda_param) * redundancy
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate is None:
                break
            selected.append(best_candidate)

        return selected[:limit]

    def _compose_group(
        self,
        indices: Sequence[int],
        records: Sequence[QARecord],
        recipe: str,
        extra_info: dict[str, Any],
    ) -> MultiStepQA:
        """Combine individual QA records into a multi-step QA bundle."""
        questions = [records[idx].question for idx in indices]
        answers = [records[idx].answer for idx in indices]
        doc_indices = self._merge_doc_indices(records, indices)
        info: dict[str, Any] = {
            "recipe": recipe,
            "source_ids": [records[idx].source_id for idx in indices],
            "group_size": len(indices),
        }
        info.update(extra_info)
        return MultiStepQA(
            questions=questions,
            answers=answers,
            doc_indices=doc_indices,
            info=info,
        )

    @staticmethod
    def _merge_doc_indices(
        records: Sequence[QARecord],
        indices: Sequence[int],
    ) -> list[int]:
        """Deduplicate document indices across grouped QA records."""
        merged = list(
            dict.fromkeys(
                idx
                for idx in chain.from_iterable(records[i].doc_indices for i in indices)
            )
        )
        return merged


PIPELINE = MultiStepQAPipeline
