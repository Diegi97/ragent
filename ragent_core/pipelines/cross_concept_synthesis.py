from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm

from ragent_core.pipelines.base import BasePipeline, PipelineMetadata
from ragent_core.pipelines.explorer_agent.explorer_agent_prompts import (
    extract_synthesized_qa,
)
from ragent_core.pipelines.explorer_agent.gemini_client import GeminiClient
from ragent_core.pipelines.prompts.cross_concept_prompts import (
    CROSS_CONCEPT_SYNTHESIS_PROMPT,
)
from ragent_core.types import QA

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL_ID = "gemini-3-flash-preview"
DEFAULT_EMBEDDING_MODEL_NAME = "MongoDB/mdbr-leaf-mt"
DEFAULT_MAX_CONCURRENT_REQUESTS = 25
DEFAULT_MIN_GROUP_SIZE = 2
DEFAULT_MAX_GROUP_SIZE = 5
DEFAULT_MAX_GROUPS = 50
DEFAULT_QAS_PER_CONCEPT = 1
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_RANDOM_SEED = 42


@dataclass
class CrossConceptSynthesisConfig:
    input_path: str
    output_path: str | None = None
    model_id: str = DEFAULT_GEMINI_MODEL_ID
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE
    max_groups: int = DEFAULT_MAX_GROUPS
    qas_per_concept: int = DEFAULT_QAS_PER_CONCEPT
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    seed: int = DEFAULT_RANDOM_SEED


class _NoOpRetriever:
    def search_tool(self, queries: list[str]) -> str:
        return "<search_results></search_results>"

    def text_scan_tool(
        self,
        pattern: str,
        fixed_string: bool = True,
        case_sensitive: bool = False,
        max_results: int = 25,
        snippet_chars: int = 200,
    ) -> str:
        return ""

    def read_tool(self, doc_ids: list[int]) -> str:
        return ""


class CrossConceptSynthesisPipeline(BasePipeline):
    Config = CrossConceptSynthesisConfig
    metadata = PipelineMetadata(
        name="qa/cross_concept_synthesis",
        description="Pipeline for cross-concept QA synthesis from existing QA pairs.",
    )

    def __init__(self) -> None:
        super().__init__()
        self._client = GeminiClient(_NoOpRetriever())
        self._embedding_model: Optional[SentenceTransformer] = None

    def _load_input(self, path: str) -> tuple[dict[str, Any] | list[Any], list[dict]]:
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSON not found: {input_path}")
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "qas" in data:
            raw_qas = data["qas"]
        elif isinstance(data, list):
            raw_qas = data
        else:
            raise ValueError(
                "Input JSON must be a list of QA records or a dict with a 'qas' field."
            )

        if not isinstance(raw_qas, list):
            raise ValueError("The 'qas' field must be a list.")

        return data, raw_qas

    @staticmethod
    def _qa_from_record(record: dict[str, Any]) -> Optional[QA]:
        question = record.get("question")
        answer = record.get("answer")
        if not question or not answer:
            return None
        return QA(
            question=question,
            answer=answer,
            doc_ids=record.get("doc_ids", []),
            info=record.get("info", {}) or {},
        )

    @staticmethod
    def _qa_to_record(qa: QA) -> dict[str, Any]:
        return {
            "question": qa.question,
            "answer": qa.answer,
            "doc_ids": list(qa.doc_ids),
            "info": qa.info,
        }

    def _get_embedding_model(self, model_name: str) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info("Loading embedding model %s", model_name)
            self._embedding_model = SentenceTransformer(model_name)
        return self._embedding_model

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _build_concept_groups(
        self,
        concepts: list[str],
        embeddings: np.ndarray,
        min_group_size: int,
        max_group_size: int,
        similarity_threshold: float,
        max_groups: int,
        rng: random.Random,
    ) -> list[list[int]]:
        if len(concepts) < min_group_size:
            return []

        embeddings = self._normalize_embeddings(embeddings)
        similarity = embeddings @ embeddings.T
        indices = list(range(len(concepts)))
        rng.shuffle(indices)

        groups: list[list[int]] = []
        used: set[int] = set()
        for idx in indices:
            if idx in used:
                continue
            target_size = rng.randint(min_group_size, max_group_size)
            candidates = [
                j
                for j in indices
                if j != idx
                and j not in used
                and similarity[idx, j] >= similarity_threshold
            ]
            candidates.sort(key=lambda j: similarity[idx, j], reverse=True)
            group = [idx] + candidates[: target_size - 1]
            if len(group) < target_size:
                continue
            for j in group:
                used.add(j)
            groups.append(group)
            if max_groups and len(groups) >= max_groups:
                break

        return groups

    @staticmethod
    def _format_concept_group(concepts: Iterable[str], qas_by_concept: dict[str, list[QA]]) -> str:
        lines = ["<concepts>"]
        for concept in concepts:
            lines.append("  <concept>")
            lines.append(f"    <name>{concept}</name>")
            lines.append("    <qa_pairs>")
            for qa in qas_by_concept.get(concept, []):
                lines.append("      <qa>")
                lines.append(f"        <question>{qa.question.strip()}</question>")
                lines.append(f"        <answer>{qa.answer.strip()}</answer>")
                lines.append("      </qa>")
            lines.append("    </qa_pairs>")
            lines.append("  </concept>")
        lines.append("</concepts>")
        return "\n".join(lines)

    async def _synthesize_group(
        self,
        concept_group: list[str],
        qas_by_concept: dict[str, list[QA]],
        model_id: str,
        semaphore: asyncio.Semaphore,
    ) -> Optional[QA]:
        prompt = self._format_concept_group(concept_group, qas_by_concept)
        response = await self._client.chat_completion_with_retry(
            content=prompt,
            model=model_id,
            semaphore=semaphore,
            system_prompt=CROSS_CONCEPT_SYNTHESIS_PROMPT,
            use_tools=False,
        )
        query, answer = extract_synthesized_qa(response)
        if not query or not answer or query == "NONE" or answer == "NONE":
            return None

        merged_doc_ids: set[int] = set()
        source_qas: list[dict[str, Any]] = []
        for concept in concept_group:
            for qa in qas_by_concept.get(concept, []):
                merged_doc_ids.update(qa.doc_ids or [])
                source_qas.append(
                    {
                        "concept": qa.info.get("concept"),
                        "question": qa.question,
                        "answer": qa.answer,
                        "doc_ids": list(qa.doc_ids),
                    }
                )

        info = {
            "cross_concept_synthesis": True,
            "source_concepts": list(concept_group),
            "source_qa_pairs": source_qas,
        }
        return QA(
            question=query,
            answer=answer,
            doc_ids=sorted(merged_doc_ids),
            info=info,
        )

    async def generate(self, config: CrossConceptSynthesisConfig) -> list[QA]:
        data, raw_qas = self._load_input(config.input_path)
        parsed_qas = [self._qa_from_record(record) for record in raw_qas]
        parsed_qas = [qa for qa in parsed_qas if qa is not None]

        if not parsed_qas:
            raise ValueError("No valid QA pairs found in input.")

        rng = random.Random(config.seed)

        concept_to_qas: dict[str, list[QA]] = {}
        for qa in parsed_qas:
            concept = qa.info.get("concept")
            if not concept:
                continue
            concept_to_qas.setdefault(concept, []).append(qa)

        concepts = list(concept_to_qas)
        if len(concepts) < config.min_group_size:
            logger.warning(
                "Not enough concepts (%d) to form groups of at least %d.",
                len(concepts),
                config.min_group_size,
            )
            return parsed_qas

        embedding_model = self._get_embedding_model(config.embedding_model_name)
        embeddings = embedding_model.encode(concepts)

        groups = self._build_concept_groups(
            concepts=concepts,
            embeddings=np.array(embeddings),
            min_group_size=config.min_group_size,
            max_group_size=config.max_group_size,
            similarity_threshold=config.similarity_threshold,
            max_groups=config.max_groups,
            rng=rng,
        )

        if not groups:
            logger.warning("No concept groups created for cross-concept synthesis.")
            return parsed_qas

        qas_by_concept: dict[str, list[QA]] = {}
        for concept, qa_list in concept_to_qas.items():
            if config.qas_per_concept <= 0:
                selected = qa_list
            elif len(qa_list) <= config.qas_per_concept:
                selected = qa_list
            else:
                selected = rng.sample(qa_list, k=config.qas_per_concept)
            qas_by_concept[concept] = selected

        semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        tasks = [
            self._synthesize_group(
                [concepts[i] for i in group],
                qas_by_concept,
                config.model_id,
                semaphore,
            )
            for group in groups
        ]
        results = await tqdm.gather(
            *tasks,
            desc="Cross-concept synthesis",
            total=len(tasks),
        )

        # Filter out existing questions and none questions from the previous synthesis step
        existing_questions = {qa.question.strip().lower() for qa in parsed_qas}
        new_qas = []
        for qa in results:
            if qa is None:
                continue
            if qa.question.strip().lower() in existing_questions:
                continue
            existing_questions.add(qa.question.strip().lower())
            new_qas.append(qa)

        if isinstance(data, dict) and "qas" in data:
            merged_records = list(raw_qas)
            merged_records.extend(self._qa_to_record(qa) for qa in new_qas)
            data["qas"] = merged_records
        else:
            data = list(raw_qas)
            data.extend(self._qa_to_record(qa) for qa in new_qas)

        output_path = (
            Path(config.output_path)
            if config.output_path
            else Path(config.input_path).with_name(
                f"{Path(config.input_path).stem}_cross_concept.json"
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            "Cross-concept synthesis complete: %d new QA pairs, output saved to %s",
            len(new_qas),
            output_path,
        )

        merged_qas = parsed_qas + new_qas
        return merged_qas


PIPELINE = CrossConceptSynthesisPipeline
