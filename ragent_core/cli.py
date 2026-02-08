import asyncio
import json
import logging
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import fire
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from ragent_core.pipelines.atomic import (
    DEFAULT_ANSWER_MODEL_ID,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NUM_PAIRS as DEFAULT_NUM_PAIRS_DEFAULT,
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_MAX_CONCEPTS_PER_DOC_CALL,
    DEFAULT_MAX_ANSWER_DOCS,
    DEFAULT_QUESTION_MODEL_ID,
    DEFAULT_RANDOM_SEED,
    DEFAULT_CHECKPOINT_INTERVAL as DEFAULT_CHECKPOINT_INTERVAL_ATOMIC,
    AtomicQAConfig,
    AtomicQAPipeline,
)
from ragent_core.pipelines.multistep import (
    DEFAULT_BREADTH_WEIGHT,
    DEFAULT_UNRELATED_WEIGHT,
    DEFAULT_ANSWER_MODEL_ID as DEFAULT_MULTISTEP_ANSWER_MODEL_ID,
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_FAR_QUANTILE,
    DEFAULT_MAX_CONCURRENT_REQUESTS as DEFAULT_MAX_CONCURRENT_REQUESTS_MULTISTEP,
    DEFAULT_MEDIUM_HIGH_QUANTILE,
    DEFAULT_MEDIUM_LOW_QUANTILE,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_NEAR_QUANTILE,
    DEFAULT_NUM_PAIRS as DEFAULT_NUM_PAIRS_MULTISTEP,
    DEFAULT_QUESTION_MODEL_ID as DEFAULT_MULTISTEP_QUESTION_MODEL_ID,
    DEFAULT_RANDOM_WALK_WEIGHT,
    DEFAULT_ATOMIC_WEIGHT,
    DEFAULT_CHECKPOINT_INTERVAL as DEFAULT_CHECKPOINT_INTERVAL_MULTISTEP,
    MultiStepQAConfig,
    MultiStepQAPipeline,
)
from ragent_core.pipelines.explorer_agent.explorer_agent import (
    DEFAULT_LLM_CLIENT,
    GEMINI_CONCEPT_MODEL_ID,
    GEMINI_BREADTH_MODEL_ID,
    GEMINI_DEPTH_MODEL_ID,
    GEMINI_SYNTHESIS_MODEL_ID,
    DEEPSEEK_CONCEPT_MODEL_ID,
    DEEPSEEK_BREADTH_MODEL_ID,
    DEEPSEEK_DEPTH_MODEL_ID,
    DEEPSEEK_SYNTHESIS_MODEL_ID,
    XIAOMI_CONCEPT_MODEL_ID,
    XIAOMI_BREADTH_MODEL_ID,
    XIAOMI_DEPTH_MODEL_ID,
    XIAOMI_SYNTHESIS_MODEL_ID,
    DEFAULT_MAX_CONCURRENT_REQUESTS as DEFAULT_MAX_CONCURRENT_REQUESTS_EXPLORER_AGENT,
    DEFAULT_ENABLE_TRAIN_EVAL_SPLIT,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_EVAL_SIZE,
    DEFAULT_SEED,
    DEFAULT_NUM_PAIRS as DEFAULT_NUM_PAIRS_EXPLORER_AGENT,
    DEFAULT_CHECKPOINT_INTERVAL as DEFAULT_CHECKPOINT_INTERVAL_EXPLORER_AGENT,
    ExplorerAgentQAConfig,
    ExplorerAgentQAPipeline,
)
from ragent_core.pipelines.cross_concept_synthesis import (
    DEFAULT_GEMINI_MODEL_ID as DEFAULT_CROSS_CONCEPT_MODEL_ID,
    DEFAULT_EMBEDDING_MODEL_NAME as DEFAULT_CROSS_CONCEPT_EMBEDDING_MODEL,
    DEFAULT_MAX_CONCURRENT_REQUESTS as DEFAULT_MAX_CONCURRENT_REQUESTS_CROSS,
    DEFAULT_MIN_GROUP_SIZE as DEFAULT_MIN_GROUP_SIZE_CROSS,
    DEFAULT_MAX_GROUP_SIZE as DEFAULT_MAX_GROUP_SIZE_CROSS,
    DEFAULT_MAX_GROUPS as DEFAULT_MAX_GROUPS_CROSS,
    DEFAULT_QAS_PER_CONCEPT as DEFAULT_QAS_PER_CONCEPT_CROSS,
    DEFAULT_SIMILARITY_THRESHOLD as DEFAULT_SIMILARITY_THRESHOLD_CROSS,
    CrossConceptSynthesisConfig,
    CrossConceptSynthesisPipeline,
)
from ragent_core.pipelines.entity_fact_qa import (
    DEFAULT_COMPLEX_PAIR_RATIO as DEFAULT_ENTITY_FACT_COMPLEX_PAIR_RATIO,
    DEFAULT_CONCEPT_MODEL_ID as DEFAULT_ENTITY_FACT_CONCEPT_MODEL_ID,
    DEFAULT_EVAL_SIZE as DEFAULT_ENTITY_FACT_EVAL_SIZE,
    DEFAULT_FACT_MODEL_ID as DEFAULT_ENTITY_FACT_MODEL_ID,
    DEFAULT_FACT_REFINEMENT_CHUNK_SIZE as DEFAULT_ENTITY_FACT_REFINEMENT_CHUNK_SIZE,
    DEFAULT_LLM_CLIENT as DEFAULT_ENTITY_FACT_LLM_CLIENT,
    DEFAULT_MAX_CONCURRENT_REQUESTS as DEFAULT_MAX_CONCURRENT_REQUESTS_ENTITY_FACT,
    DEFAULT_MAX_QA_GENERATION_ATTEMPTS as DEFAULT_ENTITY_FACT_MAX_QA_ATTEMPTS,
    DEFAULT_MAX_DOCS_PER_ENTITY as DEFAULT_ENTITY_FACT_MAX_DOCS_PER_ENTITY,
    DEFAULT_NUM_ENTITIES as DEFAULT_ENTITY_FACT_NUM_ENTITIES,
    DEFAULT_QA_MODEL_ID as DEFAULT_ENTITY_FACT_QA_MODEL_ID,
    DEFAULT_QA_PAIRS_PER_ENTITY as DEFAULT_ENTITY_FACT_QA_PAIRS_PER_ENTITY,
    DEFAULT_SEED as DEFAULT_ENTITY_FACT_SEED,
    DEFAULT_TRAIN_SIZE as DEFAULT_ENTITY_FACT_TRAIN_SIZE,
    EntityFactMemoryRecord,
    EntityFactQAConfig,
    EntityFactQAPipeline,
)
from ragent_core.pipelines.prompts.entity_fact_prompts import ExtractedFact
from ragent_core.types import QA


def _configure_logging() -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def _qa_to_dict(qa: Any, idx: int) -> dict:
    record: dict[str, Any] = {
        "id": idx,
        "doc_ids": list(qa.doc_ids),
        "num_docs": len(qa.doc_ids),
    }

    if hasattr(qa, "questions"):
        questions = list(qa.questions)
        answers = list(qa.answers)
        record.update(
            {
                "questions": questions,
                "answers": answers,
                "num_questions": len(questions),
                "num_answers": len(answers),
            }
        )
    else:
        record.update(
            {
                "question": qa.question,
                "answer": qa.answer,
            }
        )

    info = qa.info
    if info:
        record["info"] = info

    return record


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "run"


def _generate_run_identifier(pipeline_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pipeline_slug = _slugify(pipeline_name.replace("/", "-"))
    return f"{pipeline_slug}-{timestamp}"


def _infer_dataset_id(source_path: str) -> str:
    path = Path(source_path)
    metadata_path = path.with_name("metadata.json")
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        dataset_id = metadata.get("dataset_id")
        if dataset_id:
            return str(dataset_id)
    parent_name = path.parent.name
    if parent_name:
        return parent_name
    return path.stem or "dataset"


def _resolve_dataset_id_from_config(config: Any) -> str:
    candidate = getattr(config, "data_source", None)
    if candidate:
        return str(candidate)
    source_path = getattr(config, "source_data_path", None)
    if source_path:
        return _infer_dataset_id(str(source_path))
    return "dataset"


def _resolve_output_paths(
    dataset_id: str,
    pipeline_name: str,
    output_path: Optional[str],
) -> tuple[Path, Path, str]:
    base_dir = Path("data")
    if output_path:
        target = Path(output_path)
        if target.suffix:
            run_dir = target.parent
            data_path = target
        else:
            run_dir = target
            data_path = run_dir / "data.json"
        run_identifier = run_dir.name
    else:
        run_identifier = _generate_run_identifier(pipeline_name)
        run_dir = base_dir / dataset_id / run_identifier
        data_path = run_dir / "data.json"
    metadata_path = run_dir / "metadata.json"
    return data_path, metadata_path, run_identifier


def _save_output(
    qas: list[QA],
    data_path: Path,
    metadata_path: Path,
    metadata: dict,
) -> tuple[Path, Path]:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [_qa_to_dict(qa, i + 1) for i, qa in enumerate(qas)]
    with data_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return data_path, metadata_path


def _save_split_output(
    train_qas: list[QA],
    eval_qas: list[QA],
    data_path: Path,
    metadata_path: Path,
    metadata: dict,
) -> tuple[Path, Path, Path]:
    run_dir = data_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_path.with_name("train.json")
    eval_path = data_path.with_name("eval.json")

    train_serializable = [_qa_to_dict(qa, i + 1) for i, qa in enumerate(train_qas)]
    eval_serializable = [_qa_to_dict(qa, i + 1) for i, qa in enumerate(eval_qas)]

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_serializable, f, ensure_ascii=False, indent=2)
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_serializable, f, ensure_ascii=False, indent=2)

    metadata = dict(metadata)
    metadata.setdefault("output", {})
    metadata["output"] = {
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "metadata_path": str(metadata_path),
    }
    metadata["train_pairs"] = len(train_qas)
    metadata["eval_pairs"] = len(eval_qas)
    metadata["num_pairs"] = len(train_qas) + len(eval_qas)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return train_path, eval_path, metadata_path


def _entity_fact_record_to_dict(record: EntityFactMemoryRecord) -> dict[str, Any]:
    return {
        "entity_name": record.entity_name,
        "data_source": record.data_source,
        "concept_doc_id": int(record.concept_doc_id),
        "concept_importance": record.concept_importance,
        "entity_doc_ids": [int(doc_id) for doc_id in record.entity_doc_ids],
        "facts": [
            {
                "fact_id": int(fact.fact_id),
                "statement": fact.statement,
                "doc_ids": [int(doc_id) for doc_id in fact.doc_ids],
            }
            for fact in record.facts
        ],
    }


def _entity_fact_record_from_dict(payload: dict[str, Any]) -> EntityFactMemoryRecord:
    facts_payload = payload.get("facts", [])
    if not isinstance(facts_payload, list):
        raise ValueError("Expected 'facts' to be a list in entity fact payload")

    facts = [
        ExtractedFact(
            statement=str(item.get("statement", "")),
            doc_ids=[int(doc_id) for doc_id in item.get("doc_ids", [])],
            fact_id=int(item.get("fact_id", 0)),
        )
        for item in facts_payload
        if isinstance(item, dict)
    ]
    return EntityFactMemoryRecord(
        entity_name=str(payload.get("entity_name", "")),
        data_source=str(payload.get("data_source", "")),
        concept_doc_id=int(payload.get("concept_doc_id", 0)),
        concept_importance=str(payload.get("concept_importance", "")),
        entity_doc_ids=[int(doc_id) for doc_id in payload.get("entity_doc_ids", [])],
        facts=facts,
    )


def _resolve_entity_facts_output_paths(
    dataset_id: str, output_path: Optional[str]
) -> tuple[Path, Path, str]:
    if output_path:
        target = Path(output_path)
        if target.suffix:
            run_dir = target.parent
            entity_facts_path = target
        else:
            run_dir = target
            entity_facts_path = run_dir / "entity_facts.json"
        run_identifier = run_dir.name
    else:
        run_identifier = _generate_run_identifier("qa-entity-fact")
        run_dir = Path("data") / dataset_id / run_identifier
        entity_facts_path = run_dir / "entity_facts.json"

    metadata_path = run_dir / "metadata.json"
    return entity_facts_path, metadata_path, run_identifier


def _resolve_entity_facts_input_path(entity_facts_path: str) -> Path:
    path = Path(entity_facts_path)
    if path.is_dir():
        candidate = path / "entity_facts.json"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Expected entity facts file at '{candidate}', but it does not exist"
            )
        return candidate
    return path


def _load_entity_fact_records(
    entity_facts_path: Path,
) -> tuple[list[EntityFactMemoryRecord], dict[str, Any]]:
    if not entity_facts_path.exists():
        raise FileNotFoundError(f"Expected file at '{entity_facts_path}'")

    with entity_facts_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    metadata: dict[str, Any] = {}
    records_payload: list[Any]
    if isinstance(payload, dict):
        metadata = dict(payload.get("metadata", {}))
        records_payload = payload.get("entity_facts", [])
    elif isinstance(payload, list):
        records_payload = payload
    else:
        raise ValueError(
            "Entity facts payload must be a list or dict with an 'entity_facts' key"
        )

    if not isinstance(records_payload, list):
        raise ValueError("Expected 'entity_facts' to be a list")

    records: list[EntityFactMemoryRecord] = []
    for index, item in enumerate(records_payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid entity fact record at position {index}: expected object"
            )
        record = _entity_fact_record_from_dict(item)
        if not record.entity_name:
            continue
        if not record.facts:
            continue
        records.append(record)

    return records, metadata


def _load_json_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Expected file at '{path}'")
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, list):
        raise ValueError(
            f"Expected list payload in '{path}', received {type(payload)!r}"
        )
    return payload


def _config_asdict(config: Any, exclude: tuple[str, ...] = ()) -> dict:
    if is_dataclass(config):
        data = asdict(config)
    elif hasattr(config, "__dict__"):
        data = dict(vars(config))
    else:
        data = {}
    for key in exclude:
        data.pop(key, None)
    return data


def _collect_model_metadata(config: Any) -> dict:
    mapping = {
        "question_model_id": "question",
        "answer_model_id": "answer",
        "embedding_model": "embedding",
    }
    models: dict[str, Any] = {}
    for attr, label in mapping.items():
        value = getattr(config, attr, None)
        if value is not None:
            models[label] = value
    return models


def _run_pipeline(
    pipeline: Any,
    config: Any,
    output_path: Optional[str],
) -> dict:
    _configure_logging()
    logger = logging.getLogger(__name__)

    pipeline_name = getattr(
        getattr(pipeline, "metadata", None),
        "name",
        pipeline.__class__.__name__,
    )
    logger.info("Running QA generation pipeline '%s'", pipeline_name)

    qas = asyncio.run(pipeline.generate(config))
    logger.info("Generated %d QA pairs", len(qas))

    dataset_id = _resolve_dataset_id_from_config(config)
    data_path, metadata_path, run_identifier = _resolve_output_paths(
        dataset_id,
        pipeline_name,
        output_path,
    )

    metadata = _build_metadata_dict(
        pipeline=pipeline,
        config=config,
        qas=qas,
        dataset_id=dataset_id,
        data_path=data_path,
        metadata_path=metadata_path,
        run_identifier=run_identifier,
    )
    split_enabled = config.enable_train_eval_split

    if split_enabled:
        train_size, eval_size = config.train_size, config.eval_size
        if eval_size is None and train_size is None:
            eval_size = 0.2
            train_size = None
        train_qas, eval_qas = pipeline.split_train_eval(
            qas,
            eval_size=eval_size,
            train_size=train_size,
            seed=config.seed,
        )
        train_file, eval_file, metadata_file = _save_split_output(
            train_qas,
            eval_qas,
            data_path,
            metadata_path,
            metadata,
        )
        saved_to = str(train_file.resolve())
        eval_saved_to = str(eval_file.resolve())
        metadata_file = str(metadata_file.resolve())
        logger.info("Saved train split (%d pairs) to %s", len(train_qas), saved_to)
        logger.info("Saved eval split (%d pairs) to %s", len(eval_qas), eval_saved_to)
        logger.info("Saved pipeline metadata to %s", metadata_file)
    else:
        data_file, metadata_file = _save_output(qas, data_path, metadata_path, metadata)
        saved_to = str(data_file.resolve())
        metadata_file = str(metadata_file.resolve())
        eval_saved_to = None
        logger.info("Saved QA pairs to %s", saved_to)
        logger.info("Saved pipeline metadata to %s", metadata_file)

    return {
        "num_pairs": len(qas),
        "output_path": saved_to,
        "metadata_path": metadata_file,
        "eval_path": eval_saved_to if split_enabled else None,
        "run_identifier": run_identifier,
    }


def upload_to_hf_dataset(
    data_dir: str,
    repo_id: str,
    private: bool = True,
    commit_message: Optional[str] = None,
) -> dict:
    """
    Upload a pre-split QA dataset (train/eval JSON files) to the Hugging Face Hub.

    Args:
        data_dir: Directory containing `train.json`, `eval.json`, and optionally `metadata.json`.
        repo_id: Target dataset repository on the Hub, e.g. "org/dataset-name".
        private: Whether the dataset repository should be private. Defaults to True.
        commit_message: Commit message for the dataset upload.
    """
    _configure_logging()
    logger = logging.getLogger(__name__)

    base_dir = Path(data_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found")
    if not base_dir.is_dir():
        raise ValueError(
            f"data_dir must be a directory containing the dataset files (got {data_dir})"
        )

    train_path = base_dir / "train.json"
    eval_path = base_dir / "eval.json"
    metadata_path = base_dir / "metadata.json"

    train_records = _load_json_records(train_path)
    eval_records = _load_json_records(eval_path)

    commit_message = commit_message or (
        f"Upload dataset ({len(train_records)} train, {len(eval_records)} eval)"
    )

    logger.info(
        "Preparing DatasetDict with %d train records and %d eval records",
        len(train_records),
        len(eval_records),
    )

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "eval": Dataset.from_list(eval_records),
        }
    )

    logger.info("Pushing dataset to Hub repo '%s' (private=%s)", repo_id, private)
    dataset_dict.push_to_hub(
        repo_id,
        private=private,
        commit_message=commit_message,
    )

    metadata_uploaded = False
    if metadata_path.exists():
        logger.info("Uploading metadata artifact from %s", metadata_path)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo="metadata/metadata.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload metadata",
        )
        metadata_uploaded = True
    else:
        logger.warning(
            "metadata_path='%s not found; skipping metadata upload",
            metadata_path,
        )

    logger.info(
        "Upload complete (metadata uploaded=%s). Access at https://huggingface.co/datasets/%s",
        metadata_uploaded,
        repo_id,
    )

    return {
        "repo_id": repo_id,
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "metadata_uploaded": metadata_uploaded,
        "private": private,
    }


def _build_metadata_dict(
    pipeline: Any,
    config: Any,
    qas: list[QA],
    dataset_id: str,
    data_path: Path,
    metadata_path: Path,
    run_identifier: str,
) -> dict:
    pipeline_metadata = getattr(pipeline, "metadata", None)
    metadata: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "run_identifier": run_identifier,
        "dataset_id": dataset_id,
        "pipeline": {
            "name": getattr(pipeline_metadata, "name", pipeline.__class__.__name__),
            "description": getattr(pipeline_metadata, "description", None),
            "task": getattr(pipeline_metadata, "task", None),
        },
        "num_pairs": len(qas),
        "output": {
            "data_path": str(data_path),
            "metadata_path": str(metadata_path),
        },
    }

    if isinstance(config, AtomicQAConfig):
        metadata["models"] = {
            "question": config.question_model_id,
            "answer": config.answer_model_id,
        }
        metadata["retriever_type"] = (
            "BM25Retriever" if config.use_minimal_bm25 else "HybridRetriever"
        )
        metadata["parameters"] = {
            "refinement_chunk_size": config.refinement_chunk_size,
            "use_minimal_bm25": config.use_minimal_bm25,
            "num_pairs": config.num_pairs,
            "seed": config.seed,
            "max_concurrent_requests": config.max_concurrent_requests,
            "max_concepts_per_doc": config.max_concepts_per_doc,
            "max_answer_docs": config.max_answer_docs,
        }
        return metadata

    if isinstance(config, ExplorerAgentQAConfig):
        metadata["models"] = {
            "concept": config.concept_model_id,
            "breadth": config.breadth_model_id,
            "depth": config.depth_model_id,
            "synthesis": config.synthesis_model_id,
        }
        metadata["parameters"] = {
            "num_pairs": config.num_pairs,
            "seed": config.seed,
            "max_concurrent_requests": config.max_concurrent_requests,
            "use_bm25": config.use_bm25,
            "use_lite_retriever": config.use_lite_retriever,
        }
        if config.use_bm25:
            metadata["retriever_type"] = "BM25Retriever"
        elif config.use_lite_retriever:
            metadata["retriever_type"] = "HybridRetriever (lightweight)"
            metadata["retriever_models"] = {
                "colbert": "mixedbread-ai/mxbai-edge-colbert-v0-32m",
                "reranker": "mixedbread-ai/mxbai-rerank-xsmall-v1",
            }
        else:
            metadata["retriever_type"] = "HybridRetriever"
        return metadata

    if isinstance(config, EntityFactQAConfig):
        metadata["models"] = {
            "concept": config.concept_model_id,
            "fact_extraction": config.fact_model_id,
            "qa_generation": config.qa_model_id,
        }
        metadata["parameters"] = {
            "num_entities": config.num_entities,
            "qa_pairs_per_entity": config.qa_pairs_per_entity,
            "seed": config.seed,
            "max_concurrent_requests": config.max_concurrent_requests,
            "fact_refinement_chunk_size": config.fact_refinement_chunk_size,
            "complex_pair_ratio": config.complex_pair_ratio,
            "max_qa_generation_attempts": config.max_qa_generation_attempts,
            "max_docs_per_entity": config.max_docs_per_entity,
            "use_bm25": config.use_bm25,
            "use_lite_retriever": config.use_lite_retriever,
        }
        if config.use_bm25:
            metadata["retriever_type"] = "BM25Retriever"
        elif config.use_lite_retriever:
            metadata["retriever_type"] = "HybridRetriever (lightweight)"
            metadata["retriever_models"] = {
                "colbert": "mixedbread-ai/mxbai-edge-colbert-v0-32m",
                "reranker": "mixedbread-ai/mxbai-rerank-xsmall-v1",
            }
        else:
            metadata["retriever_type"] = "HybridRetriever"
        return metadata

    model_metadata = _collect_model_metadata(config)
    if model_metadata:
        metadata["models"] = model_metadata

    parameters = _config_asdict(
        config,
        exclude=("embedding_model", "data_source", "source_data_path"),
    )
    if parameters:
        metadata["parameters"] = parameters

    return metadata


def gen_qa(
    data_source: str,
    refinement_chunk_size: int = DEFAULT_CHUNK_SIZE,
    use_minimal_bm25: bool = False,
    num_pairs: int = DEFAULT_NUM_PAIRS_DEFAULT,
    question_model: str = DEFAULT_QUESTION_MODEL_ID,
    answer_model: str = DEFAULT_ANSWER_MODEL_ID,
    seed: int = DEFAULT_RANDOM_SEED,
    sem: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    max_concepts_per_doc: int = DEFAULT_MAX_CONCEPTS_PER_DOC_CALL,
    max_answer_docs: int = DEFAULT_MAX_ANSWER_DOCS,
    output_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL_ATOMIC,
    resume_from: Optional[str] = None,
    enable_checkpointing: bool = True,
) -> dict:
    """
    Generate QA pairs using the reference atomic QA pipeline.

    Args:
        data_source: Data source identifier (e.g. "gitlab_handbook").
        refinement_chunk_size: Number of retrieved chunks to review per QA pair.
        use_minimal_bm25: Toggle lightweight BM25 refinements instead of hybrid retrieval.
        num_pairs: Number of QA pairs to generate.
        question_model: Model used for concept extraction and question generation.
        answer_model: Model used for answer generation and (default) refinement.
        seed: Random seed controlling dataset sampling.
        sem: Upper bound on simultaneous LLM calls to avoid rate limiting.
        max_concepts_per_doc: Maximum concepts sent per LLM question-generation call.
        max_answer_docs: Maximum documents retained for answer generation iterations.
        output_path: Optional override for the persistence location. By default,
            data is written to `data/{data_source}/{run_id}/data.json` alongside
            metadata in `metadata.json`. Supplying a path ending with `.json`
            writes the data file there; supplying a directory writes `data.json`
            and `metadata.json` inside it.
        checkpoint_dir: Directory to save checkpoints. Defaults to `checkpoints/{pipeline}/{run_id}/`.
        checkpoint_interval: Number of QA pairs between automatic checkpoints.
        resume_from: Path to a checkpoint file to resume from.
        enable_checkpointing: Whether to enable automatic checkpointing.
    """
    pipeline = AtomicQAPipeline()
    config = AtomicQAConfig(
        data_source=data_source,
        refinement_chunk_size=refinement_chunk_size,
        use_minimal_bm25=use_minimal_bm25,
        num_pairs=num_pairs,
        question_model_id=question_model,
        answer_model_id=answer_model,
        seed=seed,
        max_concurrent_requests=sem,
        max_concepts_per_doc=max_concepts_per_doc,
        max_answer_docs=max_answer_docs,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        resume_from=resume_from,
        enable_checkpointing=enable_checkpointing,
    )
    return _run_pipeline(pipeline, config, output_path)


def compose_multistep(
    source_data_path: str,
    data_source: Optional[str] = None,
    output_path: Optional[str] = None,
    num_pairs: int = DEFAULT_NUM_PAIRS_MULTISTEP,
    breadth_weight: float = DEFAULT_BREADTH_WEIGHT,
    random_walk_weight: float = DEFAULT_RANDOM_WALK_WEIGHT,
    unrelated_weight: float = DEFAULT_UNRELATED_WEIGHT,
    atomic_weight: float = DEFAULT_ATOMIC_WEIGHT,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL_ID,
    question_model: str = DEFAULT_MULTISTEP_QUESTION_MODEL_ID,
    answer_model: str = DEFAULT_MULTISTEP_ANSWER_MODEL_ID,
    sem: int = DEFAULT_MAX_CONCURRENT_REQUESTS_MULTISTEP,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    seed: int = DEFAULT_RANDOM_SEED,
    near_quantile: float = DEFAULT_NEAR_QUANTILE,
    far_quantile: float = DEFAULT_FAR_QUANTILE,
    medium_low_quantile: float = DEFAULT_MEDIUM_LOW_QUANTILE,
    medium_high_quantile: float = DEFAULT_MEDIUM_HIGH_QUANTILE,
    train_size: float | int | None = 0.9,
    eval_size: float | int | None = 0.1,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL_MULTISTEP,
    resume_from: Optional[str] = None,
    enable_checkpointing: bool = True,
) -> dict:
    """
    Compose multi-step QA pairs by merging an existing QA dataset with embedding recipes.

    Args:
        source_data_path: Path to a JSON file generated by the atomic QA pipeline.
        data_source: Optional identifier overriding the dataset slug used for outputs.
        output_path: Optional location to write the merged QA dataset.
        embedding_model: Hugging Face model identifier used for similarity computations.
        num_pairs: Target number of multi-step QA pairs to generate.
        breadth_weight: Relative share of `num_pairs` allocated to the breadth recipe.
        random_walk_weight: Relative share allocated to the random walk recipe.
        unrelated_weight: Relative share allocated to the multi-topic bundle recipe.
        atomic_weight: Relative share allocated to the atomic replay recipe.
        question_model: LLM used to compose the final multi-step question.
        answer_model: LLM used to synthesize the final multi-step answer.
        sem: Maximum concurrent LLM calls.
        mmr_lambda: Relevance-redundancy trade-off for breadth-first MMR selection.
        seed: Random seed for reproducible sampling.
        near_quantile: Quantile defining the NEAR distance band.
        far_quantile: Quantile defining the FAR distance band.
        medium_low_quantile: Lower bound for the MEDIUM distance band.
        medium_high_quantile: Upper bound for the MEDIUM distance band.
        checkpoint_dir: Directory to save checkpoints. Defaults to `checkpoints/{pipeline}/{run_id}/`.
        checkpoint_interval: Number of QA pairs between automatic checkpoints.
        resume_from: Path to a checkpoint file to resume from.
        enable_checkpointing: Whether to enable automatic checkpointing.
    """
    dataset_id = data_source or _infer_dataset_id(source_data_path)
    pipeline = MultiStepQAPipeline()
    config = MultiStepQAConfig(
        data_source=dataset_id,
        source_data_path=source_data_path,
        num_pairs=num_pairs,
        embedding_model=embedding_model,
        seed=seed,
        breadth_weight=breadth_weight,
        random_walk_weight=random_walk_weight,
        unrelated_weight=unrelated_weight,
        atomic_weight=atomic_weight,
        question_model_id=question_model,
        answer_model_id=answer_model,
        max_concurrent_requests=sem,
        mmr_lambda=mmr_lambda,
        near_quantile=near_quantile,
        far_quantile=far_quantile,
        medium_low_quantile=medium_low_quantile,
        medium_high_quantile=medium_high_quantile,
        train_size=train_size,
        eval_size=eval_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        resume_from=resume_from,
        enable_checkpointing=enable_checkpointing,
    )
    return _run_pipeline(pipeline, config, output_path)


def gen_explorer_agent(
    data_source: str,
    num_pairs: int = DEFAULT_NUM_PAIRS_EXPLORER_AGENT,
    seed: int = DEFAULT_SEED,
    llm: str = DEFAULT_LLM_CLIENT,
    concept_model: Optional[str] = None,
    breadth_model: Optional[str] = None,
    depth_model: Optional[str] = None,
    synthesis_model: Optional[str] = None,
    sem: int = DEFAULT_MAX_CONCURRENT_REQUESTS_EXPLORER_AGENT,
    output_path: Optional[str] = None,
    enable_train_eval_split: bool = DEFAULT_ENABLE_TRAIN_EVAL_SPLIT,
    train_size: float | int | None = DEFAULT_TRAIN_SIZE,
    eval_size: float | int | None = DEFAULT_EVAL_SIZE,
    use_bm25: bool = False,
    use_lite_retriever: bool = False,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL_EXPLORER_AGENT,
    resume_from: Optional[str] = None,
) -> dict:
    """
    Generate QA pairs using explorer agent reasoning with breadth, depth, and synthesis agents.

    Args:
        data_source: Data source identifier (e.g. "gitlab_handbook").
        num_pairs: Number of QA pairs to generate.
        seed: Random seed for reproducible generation.
        llm: LLM client to use ('gemini', 'deepseek', or 'xiaomi'). Defaults to 'gemini'.
        concept_model: Model used for concept agent reasoning. If not specified, defaults
            to the appropriate model for the selected LLM client.
        breadth_model: Model used for breadth agent reasoning. If not specified, defaults
            to the appropriate model for the selected LLM client.
        depth_model: Model used for depth agent reasoning. If not specified, defaults
            to the appropriate model for the selected LLM client.
        synthesis_model: Model used for synthesis agent reasoning. If not specified, defaults
            to the appropriate model for the selected LLM client.
        sem: Maximum concurrent LLM calls to avoid rate limiting.
        output_path: Optional override for the persistence location. By default,
            data is written to `data/{data_source}/{run_id}/data.json` alongside
            metadata in `metadata.json`. Supplying a path ending with `.json`
            writes the data file there; supplying a directory writes `data.json`
            and `metadata.json` inside it.
        enable_train_eval_split: Whether to split the generated QA pairs into train/eval sets.
        train_size: Size of the training split (float for fraction, int for absolute count).
        eval_size: Size of the eval split (float for fraction, int for absolute count).
        use_bm25: If True, uses BM25Retriever. If False (default), uses HybridRetriever.
        use_lite_retriever: If True, uses lightweight models for HybridRetriever
            (mixedbread-ai/mxbai-edge-colbert-v0-32m and mixedbread-ai/mxbai-rerank-xsmall-v1).
            Only applies when use_bm25 is False.
        checkpoint_dir: Directory to save checkpoints. Defaults to `checkpoints/{pipeline}/{run_id}/`.
        checkpoint_interval: Number of QA pairs between automatic checkpoints.
        resume_from: Path to a checkpoint file to resume from.
    """

    # Set model defaults based on LLM client if not explicitly provided
    llm_lower = llm.strip().lower()
    if llm_lower == "deepseek":
        concept_model = concept_model or DEEPSEEK_CONCEPT_MODEL_ID
        breadth_model = breadth_model or DEEPSEEK_BREADTH_MODEL_ID
        depth_model = depth_model or DEEPSEEK_DEPTH_MODEL_ID
        synthesis_model = synthesis_model or DEEPSEEK_SYNTHESIS_MODEL_ID
    elif llm_lower == "xiaomi":
        concept_model = concept_model or XIAOMI_CONCEPT_MODEL_ID
        breadth_model = breadth_model or XIAOMI_BREADTH_MODEL_ID
        depth_model = depth_model or XIAOMI_DEPTH_MODEL_ID
        synthesis_model = synthesis_model or XIAOMI_SYNTHESIS_MODEL_ID
    else:
        # Default to gemini models
        concept_model = concept_model or GEMINI_CONCEPT_MODEL_ID
        breadth_model = breadth_model or GEMINI_BREADTH_MODEL_ID
        depth_model = depth_model or GEMINI_DEPTH_MODEL_ID
        synthesis_model = synthesis_model or GEMINI_SYNTHESIS_MODEL_ID

    pipeline = ExplorerAgentQAPipeline()
    config = ExplorerAgentQAConfig(
        data_source=data_source,
        num_pairs=num_pairs,
        seed=seed,
        llm_client=llm,
        concept_model_id=concept_model,
        breadth_model_id=breadth_model,
        depth_model_id=depth_model,
        synthesis_model_id=synthesis_model,
        max_concurrent_requests=sem,
        enable_train_eval_split=enable_train_eval_split,
        train_size=train_size,
        eval_size=eval_size,
        use_bm25=use_bm25,
        use_lite_retriever=use_lite_retriever,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        resume_from=resume_from,
    )
    return _run_pipeline(pipeline, config, output_path)


def cross_concept_synthesis(
    input_path: str,
    output_path: Optional[str] = None,
    model_id: str = DEFAULT_CROSS_CONCEPT_MODEL_ID,
    embedding_model: str = DEFAULT_CROSS_CONCEPT_EMBEDDING_MODEL,
    sem: int = DEFAULT_MAX_CONCURRENT_REQUESTS_CROSS,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE_CROSS,
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE_CROSS,
    max_groups: int = DEFAULT_MAX_GROUPS_CROSS,
    qas_per_concept: int = DEFAULT_QAS_PER_CONCEPT_CROSS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD_CROSS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict:
    """
    Create cross-concept QA pairs by synthesizing across existing QA pairs.

    Args:
        input_path: Path to an existing QA JSON file (list or dict with "qas").
        output_path: Optional output path for the merged QA file. If omitted,
            writes to `<input_stem>_cross_concept.json` in the same directory.
        model_id: Gemini model used for synthesis.
        embedding_model: Embedding model used for concept grouping.
        sem: Maximum concurrent LLM calls.
        min_group_size: Minimum concepts per synthesis group.
        max_group_size: Maximum concepts per synthesis group.
        max_groups: Maximum number of concept groups to synthesize.
        qas_per_concept: Number of QA pairs sampled per concept.
        similarity_threshold: Minimum cosine similarity between concepts in a group.
        seed: Random seed for reproducibility.
    """
    _configure_logging()
    pipeline = CrossConceptSynthesisPipeline()
    config = CrossConceptSynthesisConfig(
        input_path=input_path,
        output_path=output_path,
        model_id=model_id,
        embedding_model_name=embedding_model,
        max_concurrent_requests=sem,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        max_groups=max_groups,
        qas_per_concept=qas_per_concept,
        similarity_threshold=similarity_threshold,
        seed=seed,
    )
    qas = asyncio.run(pipeline.generate(config))
    resolved_output = (
        output_path
        if output_path
        else str(
            Path(input_path).with_name(
                f"{Path(input_path).stem}_cross_concept.json"
            )
        )
    )
    return {
        "num_pairs": len(qas),
        "output_path": resolved_output,
    }


def gen_entity_fact(
    data_source: str,
    llm: str = DEFAULT_ENTITY_FACT_LLM_CLIENT,
    num_entities: int = DEFAULT_ENTITY_FACT_NUM_ENTITIES,
    qa_pairs_per_entity: int = DEFAULT_ENTITY_FACT_QA_PAIRS_PER_ENTITY,
    concept_model: str = DEFAULT_ENTITY_FACT_CONCEPT_MODEL_ID,
    fact_model: str = DEFAULT_ENTITY_FACT_MODEL_ID,
    qa_model: str = DEFAULT_ENTITY_FACT_QA_MODEL_ID,
    seed: int = DEFAULT_ENTITY_FACT_SEED,
    sem: int = DEFAULT_MAX_CONCURRENT_REQUESTS_ENTITY_FACT,
    fact_refinement_chunk_size: int = DEFAULT_ENTITY_FACT_REFINEMENT_CHUNK_SIZE,
    complex_pair_ratio: float = DEFAULT_ENTITY_FACT_COMPLEX_PAIR_RATIO,
    max_qa_generation_attempts: int = DEFAULT_ENTITY_FACT_MAX_QA_ATTEMPTS,
    max_docs_per_entity: int = DEFAULT_ENTITY_FACT_MAX_DOCS_PER_ENTITY,
    output_path: Optional[str] = None,
    enable_train_eval_split: bool = False,
    train_size: float | int | None = DEFAULT_ENTITY_FACT_TRAIN_SIZE,
    eval_size: float | int | None = DEFAULT_ENTITY_FACT_EVAL_SIZE,
    use_bm25: bool = False,
    use_lite_retriever: bool = False,
) -> dict:
    """
    Generate entity-grounded QA pairs via iterative fact extraction/refinement.

    Args:
        data_source: Data source identifier.
        llm: LLM client to use ('gemini' or 'openai').
        num_entities: Number of sampled entities/concepts.
        qa_pairs_per_entity: Number of QA pairs to generate for each entity.
        concept_model: Model used for entity sampling.
        fact_model: Model used for iterative fact extraction/refinement.
        qa_model: Model used for QA generation from facts.
        seed: Random seed.
        sem: Maximum concurrent LLM calls.
        fact_refinement_chunk_size: Number of documents processed per refinement step.
        complex_pair_ratio: Fraction of QA pairs that should be complex.
        max_qa_generation_attempts: Retries for QA generation per entity.
        max_docs_per_entity: Maximum number of documents to retrieve per entity.
        output_path: Optional output file/directory override.
        enable_train_eval_split: Whether to save train/eval splits.
        train_size: Train split size if splitting is enabled.
        eval_size: Eval split size if splitting is enabled.
        use_bm25: If True, use BM25 retriever.
        use_lite_retriever: If True, use lightweight HybridRetriever models.
    """
    pipeline = EntityFactQAPipeline()
    config = EntityFactQAConfig(
        data_source=data_source,
        llm_client=llm,
        num_entities=num_entities,
        qa_pairs_per_entity=qa_pairs_per_entity,
        concept_model_id=concept_model,
        fact_model_id=fact_model,
        qa_model_id=qa_model,
        seed=seed,
        max_concurrent_requests=sem,
        fact_refinement_chunk_size=fact_refinement_chunk_size,
        complex_pair_ratio=complex_pair_ratio,
        max_qa_generation_attempts=max_qa_generation_attempts,
        max_docs_per_entity=max_docs_per_entity,
        enable_train_eval_split=enable_train_eval_split,
        train_size=train_size,
        eval_size=eval_size,
        use_bm25=use_bm25,
        use_lite_retriever=use_lite_retriever,
    )
    return _run_pipeline(pipeline, config, output_path)


def gen_entity_fact_memory(
    data_source: str,
    llm: str = DEFAULT_ENTITY_FACT_LLM_CLIENT,
    num_entities: int = DEFAULT_ENTITY_FACT_NUM_ENTITIES,
    concept_model: str = DEFAULT_ENTITY_FACT_CONCEPT_MODEL_ID,
    fact_model: str = DEFAULT_ENTITY_FACT_MODEL_ID,
    seed: int = DEFAULT_ENTITY_FACT_SEED,
    sem: int = DEFAULT_MAX_CONCURRENT_REQUESTS_ENTITY_FACT,
    fact_refinement_chunk_size: int = DEFAULT_ENTITY_FACT_REFINEMENT_CHUNK_SIZE,
    max_docs_per_entity: int = DEFAULT_ENTITY_FACT_MAX_DOCS_PER_ENTITY,
    output_path: Optional[str] = None,
    use_bm25: bool = False,
    use_lite_retriever: bool = False,
) -> dict:
    """
    Generate and persist entity concepts + grounded facts for later QA generation.

    Args:
        data_source: Data source identifier.
        llm: LLM client to use ('gemini' or 'openai').
        num_entities: Number of sampled entities/concepts.
        concept_model: Model used for entity sampling.
        fact_model: Model used for iterative fact extraction/refinement.
        seed: Random seed.
        sem: Maximum concurrent LLM calls.
        fact_refinement_chunk_size: Number of documents processed per refinement step.
        max_docs_per_entity: Maximum number of documents to retrieve per entity.
        output_path: Optional output file/directory override.
        use_bm25: If True, use BM25 retriever.
        use_lite_retriever: If True, use lightweight HybridRetriever models.
    """
    _configure_logging()
    logger = logging.getLogger(__name__)

    pipeline = EntityFactQAPipeline()
    config = EntityFactQAConfig(
        data_source=data_source,
        llm_client=llm,
        num_entities=num_entities,
        qa_pairs_per_entity=DEFAULT_ENTITY_FACT_QA_PAIRS_PER_ENTITY,
        concept_model_id=concept_model,
        fact_model_id=fact_model,
        qa_model_id=DEFAULT_ENTITY_FACT_QA_MODEL_ID,
        seed=seed,
        max_concurrent_requests=sem,
        fact_refinement_chunk_size=fact_refinement_chunk_size,
        complex_pair_ratio=DEFAULT_ENTITY_FACT_COMPLEX_PAIR_RATIO,
        max_qa_generation_attempts=DEFAULT_ENTITY_FACT_MAX_QA_ATTEMPTS,
        max_docs_per_entity=max_docs_per_entity,
        enable_train_eval_split=False,
        train_size=DEFAULT_ENTITY_FACT_TRAIN_SIZE,
        eval_size=DEFAULT_ENTITY_FACT_EVAL_SIZE,
        use_bm25=use_bm25,
        use_lite_retriever=use_lite_retriever,
    )

    entity_facts, data_source_description = asyncio.run(
        pipeline.generate_entity_facts(config)
    )

    entity_facts_path, metadata_path, run_identifier = _resolve_entity_facts_output_paths(
        data_source, output_path
    )
    entity_facts_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now().isoformat(),
        "run_identifier": run_identifier,
        "dataset_id": data_source,
        "pipeline": {
            "name": "qa/entity_fact_memory",
            "description": (
                "Entity concept and fact generation stage from the entity-fact pipeline"
            ),
        },
        "models": {
            "concept": concept_model,
            "fact_extraction": fact_model,
        },
        "parameters": {
            "num_entities": num_entities,
            "seed": seed,
            "max_concurrent_requests": sem,
            "fact_refinement_chunk_size": fact_refinement_chunk_size,
            "max_docs_per_entity": max_docs_per_entity,
            "use_bm25": use_bm25,
            "use_lite_retriever": use_lite_retriever,
        },
        "data_source_description": data_source_description,
        "num_entities_with_facts": len(entity_facts),
        "output": {
            "entity_facts_path": str(entity_facts_path),
            "metadata_path": str(metadata_path),
        },
    }

    with entity_facts_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "metadata": metadata,
                "entity_facts": [
                    _entity_fact_record_to_dict(record) for record in entity_facts
                ],
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    resolved_entity_facts_path = str(entity_facts_path.resolve())
    resolved_metadata_path = str(metadata_path.resolve())
    logger.info("Saved entity facts to %s", resolved_entity_facts_path)
    logger.info("Saved stage metadata to %s", resolved_metadata_path)

    return {
        "num_entities_with_facts": len(entity_facts),
        "entity_facts_path": resolved_entity_facts_path,
        "metadata_path": resolved_metadata_path,
        "run_identifier": run_identifier,
    }


def gen_entity_fact_qas(
    entity_facts_path: str,
    data_source: Optional[str] = None,
    llm: str = DEFAULT_ENTITY_FACT_LLM_CLIENT,
    qa_pairs_per_entity: int = DEFAULT_ENTITY_FACT_QA_PAIRS_PER_ENTITY,
    qa_model: str = DEFAULT_ENTITY_FACT_QA_MODEL_ID,
    seed: int = DEFAULT_ENTITY_FACT_SEED,
    sem: int = DEFAULT_MAX_CONCURRENT_REQUESTS_ENTITY_FACT,
    complex_pair_ratio: float = DEFAULT_ENTITY_FACT_COMPLEX_PAIR_RATIO,
    max_qa_generation_attempts: int = DEFAULT_ENTITY_FACT_MAX_QA_ATTEMPTS,
    output_path: Optional[str] = None,
    enable_train_eval_split: bool = False,
    train_size: float | int | None = DEFAULT_ENTITY_FACT_TRAIN_SIZE,
    eval_size: float | int | None = DEFAULT_ENTITY_FACT_EVAL_SIZE,
) -> dict:
    """
    Generate QA pairs from a previously saved entity-facts artifact.

    Args:
        entity_facts_path: Path to `entity_facts.json` or its containing directory.
        data_source: Optional data source override. Defaults to value from artifact metadata.
        llm: LLM client to use ('gemini' or 'openai').
        qa_pairs_per_entity: Number of QA pairs to generate per saved entity.
        qa_model: Model used for QA generation from facts.
        seed: Random seed.
        sem: Maximum concurrent LLM calls.
        complex_pair_ratio: Fraction of QA pairs that should be complex.
        max_qa_generation_attempts: Retries for QA generation per entity.
        output_path: Optional output file/directory override. Defaults to sibling `data.json`.
        enable_train_eval_split: Whether to save train/eval splits.
        train_size: Train split size if splitting is enabled.
        eval_size: Eval split size if splitting is enabled.
        use_bm25: If True, use BM25 retriever for pipeline runtime initialization.
        use_lite_retriever: If True, use lightweight HybridRetriever models.
    """
    _configure_logging()
    logger = logging.getLogger(__name__)

    resolved_entity_facts_path = _resolve_entity_facts_input_path(entity_facts_path)
    entity_facts, stage_metadata = _load_entity_fact_records(resolved_entity_facts_path)
    if not entity_facts:
        raise ValueError(f"No valid entity facts found in '{resolved_entity_facts_path}'")

    resolved_data_source = data_source or stage_metadata.get("dataset_id")
    if not resolved_data_source:
        resolved_data_source = entity_facts[0].data_source
    if not resolved_data_source:
        raise ValueError(
            "Unable to infer data_source. Provide --data_source explicitly."
        )

    pipeline = EntityFactQAPipeline()
    config = EntityFactQAConfig(
        data_source=str(resolved_data_source),
        llm_client=llm,
        num_entities=len(entity_facts),
        qa_pairs_per_entity=qa_pairs_per_entity,
        concept_model_id=DEFAULT_ENTITY_FACT_CONCEPT_MODEL_ID,
        fact_model_id=DEFAULT_ENTITY_FACT_MODEL_ID,
        qa_model_id=qa_model,
        seed=seed,
        max_concurrent_requests=sem,
        fact_refinement_chunk_size=DEFAULT_ENTITY_FACT_REFINEMENT_CHUNK_SIZE,
        complex_pair_ratio=complex_pair_ratio,
        max_qa_generation_attempts=max_qa_generation_attempts,
        max_docs_per_entity=DEFAULT_ENTITY_FACT_MAX_DOCS_PER_ENTITY,
        enable_train_eval_split=enable_train_eval_split,
        train_size=train_size,
        eval_size=eval_size,
    )

    qas = asyncio.run(
        pipeline.generate_qas_from_entity_facts(
            config=config,
            entity_facts=entity_facts,
            data_source_description=stage_metadata.get("data_source_description"),
        )
    )
    logger.info("Generated %d QA pairs from saved entity facts", len(qas))

    if output_path is None:
        output_path = str(resolved_entity_facts_path.with_name("data.json"))

    data_path, metadata_path, run_identifier = _resolve_output_paths(
        str(resolved_data_source),
        "qa/entity_fact",
        output_path,
    )
    metadata = _build_metadata_dict(
        pipeline=pipeline,
        config=config,
        qas=qas,
        dataset_id=str(resolved_data_source),
        data_path=data_path,
        metadata_path=metadata_path,
        run_identifier=run_identifier,
    )
    metadata["input"] = {
        "entity_facts_path": str(resolved_entity_facts_path),
        "num_entities_with_facts": len(entity_facts),
    }

    split_enabled = config.enable_train_eval_split
    if split_enabled:
        train_qas, eval_qas = pipeline.split_train_eval(
            qas,
            eval_size=config.eval_size,
            train_size=config.train_size,
            seed=config.seed,
        )
        train_file, eval_file, metadata_file = _save_split_output(
            train_qas,
            eval_qas,
            data_path,
            metadata_path,
            metadata,
        )
        saved_to = str(train_file.resolve())
        eval_saved_to = str(eval_file.resolve())
        metadata_file = str(metadata_file.resolve())
        logger.info("Saved train split (%d pairs) to %s", len(train_qas), saved_to)
        logger.info("Saved eval split (%d pairs) to %s", len(eval_qas), eval_saved_to)
        logger.info("Saved pipeline metadata to %s", metadata_file)
    else:
        data_file, metadata_file = _save_output(qas, data_path, metadata_path, metadata)
        saved_to = str(data_file.resolve())
        metadata_file = str(metadata_file.resolve())
        eval_saved_to = None
        logger.info("Saved QA pairs to %s", saved_to)
        logger.info("Saved pipeline metadata to %s", metadata_file)

    return {
        "num_pairs": len(qas),
        "output_path": saved_to,
        "metadata_path": metadata_file,
        "eval_path": eval_saved_to if split_enabled else None,
        "run_identifier": run_identifier,
        "entity_facts_path": str(resolved_entity_facts_path.resolve()),
    }


class RAGentCLI:
    def __init__(self) -> None:
        self.data = {
            "gen_qa": gen_qa,
            "compose_multistep": compose_multistep,
            "gen_explorer_agent": gen_explorer_agent,
            "gen_entity_fact": gen_entity_fact,
            "gen_entity_fact_memory": gen_entity_fact_memory,
            "gen_entity_fact_qas": gen_entity_fact_qas,
            "cross_concept_synthesis": cross_concept_synthesis,
            "upload_to_hf": upload_to_hf_dataset,
        }


def main() -> None:
    fire.Fire(RAGentCLI)


if __name__ == "__main__":
    main()
