import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from pydantic import ValidationError

from data_pipelines.pipelines.deep_search_task_generation.config import (
    DEFAULT_ENTITY_MODEL_ID,
    DEFAULT_QA_MODEL_ID,
    DEFAULT_RETRIEVER_WORKER_PORT,
    DeepSearchTaskGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.pipeline import (
    generate_deep_search_qas_flow,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.models import (
    DEFAULT_PI_MODEL,
    PiThinkingLevel,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.pipeline import (
    generate_deep_search_rubrics_flow,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.pipeline import (
    prepare_deep_search_tasks_flow,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker import (
    RetrieverWorkerConfig,
    serve_retriever_worker,
)
from data_pipelines.pipelines.retrieval_evaluation.config import (
    RetrievalEvaluationConfig,
    SearchType,
)
from data_pipelines.pipelines.retrieval_evaluation.evaluator import (
    evaluate_retrieval,
)
from data_pipelines.pipelines.search_query_generation.config import (
    GENERATOR_MODEL,
    RetrievalQueriesConfig,
)
from data_pipelines.pipelines.search_query_generation.pipeline import (
    search_query_generation_batch_flow,
)
from ragent_core.config.logging import configure_logging
from ragent_core.retrievers.retriever import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_RERANKER_MODEL_NAME,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Generate search-query-generation JSONL data with Prefect and Phoenix.",
)

evaluation_app = typer.Typer(
    no_args_is_help=True,
    help="Evaluate generated retrieval queries against a LanceDB retriever.",
)

deep_search_tasks_app = typer.Typer(
    no_args_is_help=True,
    help="Generate corpus-grounded QA and rubric tasks for deep search.",
)


@evaluation_app.callback()
def evaluation_main() -> None:
    """Retrieval evaluation commands."""


def _retrieval_queries_config(
    table_name: str,
    output_path: Path,
    lancedb_db_uri: Path,
    num_queries: int,
    hard_negatives_per_query: int,
    round_trip_top_k: int,
    candidate_mining_top_k: int,
    contrastive_candidate_count: int,
    generator_model: str,
    retriever_namespace: str,
    seed: int,
    llm_concurrency: int,
    retriever_concurrency: int,
) -> RetrievalQueriesConfig:
    return RetrievalQueriesConfig(
        table_name=table_name,
        output_path=output_path,
        lancedb_db_uri=lancedb_db_uri,
        num_queries=num_queries,
        hard_negatives_per_query=hard_negatives_per_query,
        round_trip_top_k=round_trip_top_k,
        candidate_mining_top_k=candidate_mining_top_k,
        contrastive_candidate_count=contrastive_candidate_count,
        generator_model=generator_model,
        retriever_namespace=retriever_namespace,
        seed=seed,
        llm_concurrency=llm_concurrency,
        retriever_concurrency=retriever_concurrency,
    )


def _parse_cutoffs(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter(
            "Cutoffs must be a comma-separated list of integers."
        ) from exc


@app.command()
def run(
    table_name: Annotated[str, typer.Option(help="LanceDB table prefix.")],
    output_path: Annotated[
        Path,
        typer.Option(help="Base canonical queries.jsonl path."),
    ] = RetrievalQueriesConfig.model_fields["output_path"].default,
    lancedb_db_uri: Annotated[
        Path,
        typer.Option(help="Repository-level LanceDB directory."),
    ] = RetrievalQueriesConfig.model_fields["lancedb_db_uri"].default,
    num_queries: Annotated[
        int,
        typer.Option(min=0, help="Number of unique query objects."),
    ] = 10,
    hard_negatives_per_query: Annotated[
        int,
        typer.Option(min=0),
    ] = 10,
    round_trip_top_k: Annotated[int, typer.Option(min=1)] = 50,
    candidate_mining_top_k: Annotated[int, typer.Option(min=1)] = 50,
    contrastive_candidate_count: Annotated[int, typer.Option(min=0)] = 25,
    generator_model: Annotated[str, typer.Option()] = GENERATOR_MODEL,
    retriever_namespace: Annotated[str, typer.Option()] = "default",
    seed: Annotated[int, typer.Option()] = 42,
    llm_concurrency: Annotated[int, typer.Option(min=1)] = 4,
    retriever_concurrency: Annotated[int, typer.Option(min=1)] = 2,
) -> None:
    """Run a batch immediately in the current process."""
    load_dotenv()
    config = _retrieval_queries_config(
        table_name,
        output_path,
        lancedb_db_uri,
        num_queries,
        hard_negatives_per_query,
        round_trip_top_k,
        candidate_mining_top_k,
        contrastive_candidate_count,
        generator_model,
        retriever_namespace,
        seed,
        llm_concurrency,
        retriever_concurrency,
    )
    metadata = asyncio.run(search_query_generation_batch_flow(config))
    typer.echo(json.dumps(metadata, indent=2, ensure_ascii=False))


def _deep_search_task_config(
    data_source: str,
    output_root: Path,
    entities_file: Path | None,
    num_entities: int,
    entity_model_id: str,
    seed: int,
    sample_size: int,
    num_chunks_per_entity: int,
    fact_extraction_chunks_per_request: int,
    llm_concurrency: int,
    retriever_worker_port: int,
) -> DeepSearchTaskGenerationConfig:
    return DeepSearchTaskGenerationConfig(
        data_source=data_source,
        output_root=output_root,
        entities_file=entities_file,
        num_entities=num_entities,
        entity_model_id=entity_model_id,
        seed=seed,
        sample_size=sample_size,
        num_chunks_per_entity=num_chunks_per_entity,
        fact_extraction_chunks_per_request=fact_extraction_chunks_per_request,
        llm_concurrency=llm_concurrency,
        retriever_worker_port=retriever_worker_port,
    )


@deep_search_tasks_app.command("retriever")
def run_deep_search_task_retriever(
    lancedb_db_uri: Annotated[
        Path,
        typer.Option(help="Repository-level LanceDB directory."),
    ] = RetrieverWorkerConfig.model_fields["lancedb_db_uri"].default,
    retriever_namespace: Annotated[str, typer.Option()] = "default",
    retriever_device: Annotated[str | None, typer.Option()] = None,
    rerank_threshold: Annotated[float, typer.Option(min=0.0)] = 3.0,
    port: Annotated[
        int,
        typer.Option(min=1, max=65535),
    ] = DEFAULT_RETRIEVER_WORKER_PORT,
) -> None:
    """Serve the sequential deep-search task retriever on localhost."""
    load_dotenv()
    configure_logging()
    config = RetrieverWorkerConfig(
        lancedb_db_uri=lancedb_db_uri,
        retriever_namespace=retriever_namespace,
        retriever_device=retriever_device,
        rerank_threshold=rerank_threshold,
        port=port,
    )
    serve_retriever_worker(config)


@deep_search_tasks_app.command("prepare")
def prepare_deep_search_tasks(
    data_source: Annotated[str, typer.Option(help="Corpus loader or HF dataset ID.")],
    output_root: Annotated[
        Path, typer.Option()
    ] = DeepSearchTaskGenerationConfig.model_fields["output_root"].default,
    entities_file: Annotated[
        Path | None,
        typer.Option(
            "--entities-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help=(
                "Reuse entities from an existing entities.jsonl and skip entity "
                "extraction. At most --num-entities records are loaded."
            ),
        ),
    ] = None,
    num_entities: Annotated[int, typer.Option(min=0)] = 100,
    entity_model_id: Annotated[str, typer.Option()] = DEFAULT_ENTITY_MODEL_ID,
    seed: Annotated[int, typer.Option()] = 42,
    sample_size: Annotated[int, typer.Option(min=1)] = 4,
    num_chunks_per_entity: Annotated[
        int,
        typer.Option(
            min=1,
            help=(
                "Maximum fused chunks scored by the CrossEncoder per entity; "
                "the relevance threshold may return fewer."
            ),
        ),
    ] = 100,
    fact_extraction_chunks_per_request: Annotated[int, typer.Option(min=1)] = 5,
    llm_concurrency: Annotated[int, typer.Option(min=1)] = 25,
    retriever_worker_port: Annotated[
        int,
        typer.Option(min=1, max=65535),
    ] = DEFAULT_RETRIEVER_WORKER_PORT,
) -> None:
    """Discover entities, write fact requests, and upload the input dataset."""
    load_dotenv()
    config = _deep_search_task_config(
        data_source,
        output_root,
        entities_file,
        num_entities,
        entity_model_id,
        seed,
        sample_size,
        num_chunks_per_entity,
        fact_extraction_chunks_per_request,
        llm_concurrency,
        retriever_worker_port,
    )
    metadata = asyncio.run(prepare_deep_search_tasks_flow(config))
    typer.echo(json.dumps(metadata, indent=2, ensure_ascii=False))


@deep_search_tasks_app.command("generate-qas")
def generate_deep_search_qas(
    prepare_run_directory: Annotated[
        Path,
        typer.Option(
            "--prepare-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help="Directory produced by the prepare command.",
        ),
    ],
    batch_output_dataset_name: Annotated[
        str,
        typer.Option(
            "--batch-dataset",
            help="Completed Fireworks output dataset name.",
        ),
    ],
    qa_pairs_per_entity: Annotated[
        int,
        typer.Option(
            "--qa-pairs-per-entity",
            min=0,
            help="Number of QA pairs to request for each entity.",
        ),
    ] = 4,
    qa_model_id: Annotated[
        str,
        typer.Option(
            "--qa-model-id",
            help="Model used to generate QA candidates.",
        ),
    ] = DEFAULT_QA_MODEL_ID,
    complex_pair_ratio: Annotated[
        float,
        typer.Option(
            "--complex-pair-ratio",
            min=0.0,
            max=1.0,
            help="Minimum share of generated QA pairs targeted as complex.",
        ),
    ] = 0.7,
    max_qa_generation_attempts: Annotated[
        int,
        typer.Option(
            "--max-qa-generation-attempts",
            min=1,
            help="Maximum QA generation rounds per entity.",
        ),
    ] = 4,
    llm_concurrency: Annotated[
        int,
        typer.Option(
            "--llm-concurrency",
            min=1,
            help="Maximum concurrent QA-generation LLM calls.",
        ),
    ] = 25,
    download_timeout: Annotated[
        float,
        typer.Option(
            "--download-timeout",
            min=0.001,
            help="Fireworks batch-output download timeout in seconds.",
        ),
    ] = 600.0,
) -> None:
    """Download a completed Fireworks batch and generate QA records."""
    load_dotenv()
    metadata = asyncio.run(
        generate_deep_search_qas_flow(
            prepare_run_directory=prepare_run_directory,
            batch_output_dataset_name=batch_output_dataset_name,
            qa_pairs_per_entity=qa_pairs_per_entity,
            qa_model_id=qa_model_id,
            complex_pair_ratio=complex_pair_ratio,
            max_qa_generation_attempts=max_qa_generation_attempts,
            llm_concurrency=llm_concurrency,
            download_timeout=download_timeout,
        )
    )
    typer.echo(json.dumps(metadata, indent=2, ensure_ascii=False))


@deep_search_tasks_app.command("generate-rubrics")
def generate_deep_search_rubrics(
    prepare_run_directory: Annotated[
        Path,
        typer.Option(
            "--prepare-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help="Directory produced by the prepare command.",
        ),
    ],
    batch_output_dataset_name: Annotated[
        str,
        typer.Option(
            "--batch-dataset",
            help="Completed Fireworks output dataset name.",
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="PI model used to generate question-rubric records.",
        ),
    ] = DEFAULT_PI_MODEL,
    thinking: Annotated[
        PiThinkingLevel | None,
        typer.Option(
            "--thinking",
            help="PI reasoning level; omit to use the model's default.",
        ),
    ] = None,
    num_question_rubrics: Annotated[
        int,
        typer.Option(
            "--num-rubrics",
            min=0,
            help="Total number of question-rubric records to request.",
        ),
    ] = 10,
    pi_concurrency: Annotated[
        int,
        typer.Option(
            "--pi-concurrency",
            min=1,
            help="Maximum concurrent PI subprocesses.",
        ),
    ] = 10,
    max_attempts: Annotated[
        int,
        typer.Option(
            "--max-attempts",
            min=1,
            help="Maximum PI attempts for each assigned output slot.",
        ),
    ] = 4,
    download_timeout: Annotated[
        float,
        typer.Option(
            "--download-timeout",
            min=0.001,
            help="Fireworks batch-output download timeout in seconds.",
        ),
    ] = 600.0,
) -> None:
    """Generate question-rubric records with PI from extracted entity facts."""
    load_dotenv()
    metadata = asyncio.run(
        generate_deep_search_rubrics_flow(
            prepare_run_directory=prepare_run_directory,
            batch_output_dataset_name=batch_output_dataset_name,
            model=model,
            thinking=thinking.value if thinking is not None else None,
            num_question_rubrics=num_question_rubrics,
            pi_concurrency=pi_concurrency,
            max_attempts=max_attempts,
            download_timeout=download_timeout,
        )
    )
    typer.echo(json.dumps(metadata, indent=2, ensure_ascii=False))


@evaluation_app.command("run")
def run_evaluation(
    input_directory: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Directory containing queries.jsonl and metadata.json.",
        ),
    ],
    search_type: Annotated[
        SearchType,
        typer.Option(help="LanceDB retrieval method."),
    ] = SearchType.HYBRID,
    top_k: Annotated[int, typer.Option(min=1)] = 50,
    cutoffs: Annotated[
        str,
        typer.Option(help="Strictly increasing metric cutoffs."),
    ] = "1,3,5,10,20,50",
    embedding_model: Annotated[str, typer.Option()] = DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_service_url: Annotated[
        str | None,
        typer.Option(
            help=(
                "Remote embedding service URL. Defaults to "
                "RAGENT_EMBEDDING_SERVICE_URL; when neither is set, use a "
                "local model."
            )
        ),
    ] = None,
    device: Annotated[str | None, typer.Option()] = None,
    max_seq_length: Annotated[int | None, typer.Option(min=1)] = None,
    trust_remote_code: Annotated[
        bool,
        typer.Option("--trust-remote-code/--no-trust-remote-code"),
    ] = True,
    reranker: Annotated[
        bool,
        typer.Option("--reranker/--no-reranker"),
    ] = False,
    reranker_model: Annotated[str, typer.Option()] = DEFAULT_RERANKER_MODEL_NAME,
    reranker_service_url: Annotated[
        str | None,
        typer.Option(
            help=(
                "Remote reranker service URL. Defaults to "
                "RAGENT_RERANKER_SERVICE_URL when reranking is enabled."
            )
        ),
    ] = None,
    reranker_candidate_k: Annotated[int, typer.Option(min=1)] = 50,
    reranker_threshold: Annotated[float, typer.Option()] = 0.0,
    reranker_batch_size: Annotated[int, typer.Option(min=1)] = 8,
) -> None:
    """Run one retrieval configuration against all valid generated queries."""
    load_dotenv()
    embedding_url = embedding_service_url or os.getenv("RAGENT_EMBEDDING_SERVICE_URL")
    reranker_url = reranker_service_url or os.getenv("RAGENT_RERANKER_SERVICE_URL")
    try:
        config = RetrievalEvaluationConfig(
            input_directory=input_directory,
            search_type=search_type,
            top_k=top_k,
            cutoffs=_parse_cutoffs(cutoffs),
            embedding_model=embedding_model,
            embedding_service_url=embedding_url,
            device=device,
            max_seq_length=max_seq_length,
            trust_remote_code=trust_remote_code,
            reranker=reranker,
            reranker_model=reranker_model,
            reranker_service_url=reranker_url,
            reranker_candidate_k=reranker_candidate_k,
            reranker_threshold=reranker_threshold,
            reranker_batch_size=reranker_batch_size,
        )
        summary = evaluate_retrieval(config)
    except (ValidationError, ValueError, OSError) as exc:
        typer.echo(f"Evaluation configuration or input error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        typer.echo(f"Evaluation setup failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    top_cutoff = str(max(config.cutoffs))
    compact_summary = {
        "output_directory": str(summary.output_directory),
        "successful_queries": summary.successful_queries,
        "failed_queries": summary.failed_queries,
        "coverage": summary.coverage,
        f"chunk_recall@{top_cutoff}": summary.metrics["chunk"]["cutoffs"][top_cutoff][
            "recall"
        ],
        f"chunk_mrr@{top_cutoff}": summary.metrics["chunk"]["cutoffs"][top_cutoff][
            "mrr"
        ],
        f"document_recall@{top_cutoff}": summary.metrics["document"]["cutoffs"][
            top_cutoff
        ]["recall"],
        f"document_mrr@{top_cutoff}": summary.metrics["document"]["cutoffs"][
            top_cutoff
        ]["mrr"],
    }
    typer.echo(json.dumps(compact_summary, indent=2, ensure_ascii=False))
    if summary.failed_queries:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
