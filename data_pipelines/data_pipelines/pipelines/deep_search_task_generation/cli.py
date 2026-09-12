import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from data_pipelines.pipelines.deep_search_task_generation.generate.qa.config import (
    QAGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.pipeline import (
    generate_deep_search_qas_flow,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.config import (
    PiThinkingLevel,
    RubricGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.pipeline import (
    generate_deep_search_rubrics_flow,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.config import (
    DeepSearchTaskGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.pipeline import (
    prepare_deep_search_tasks_flow,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.config import (
    RetrieverWorkerConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.server import (
    serve_retriever_worker,
)
from ragent_core.config.logging import configure_logging

PREPARE_DIRECTORY_HELP = "Directory produced by the prepare command."

app = typer.Typer(
    no_args_is_help=True,
    help="Generate corpus-grounded QA and rubric tasks for deep search.",
)


@app.command("retriever")
def run_deep_search_task_retriever(
    retriever_namespace: Annotated[
        str, typer.Option()
    ] = RetrieverWorkerConfig.model_fields["retriever_namespace"].default,
    retriever_device: Annotated[
        str | None, typer.Option()
    ] = RetrieverWorkerConfig.model_fields["retriever_device"].default,
    rerank_threshold: Annotated[
        float, typer.Option(min=0.0)
    ] = RetrieverWorkerConfig.model_fields["rerank_threshold"].default,
    port: Annotated[
        int,
        typer.Option(min=1, max=65535),
    ] = RetrieverWorkerConfig.model_fields["port"].default,
) -> None:
    """Serve the sequential deep-search task retriever on localhost."""
    load_dotenv()
    configure_logging()
    config = RetrieverWorkerConfig(
        retriever_namespace=retriever_namespace,
        retriever_device=retriever_device,
        rerank_threshold=rerank_threshold,
        port=port,
    )
    serve_retriever_worker(config)


@app.command("prepare")
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
    ] = DeepSearchTaskGenerationConfig.model_fields["entities_file"].default,
    num_entities: Annotated[
        int, typer.Option(min=0)
    ] = DeepSearchTaskGenerationConfig.model_fields["num_entities"].default,
    entity_model_id: Annotated[
        str, typer.Option()
    ] = DeepSearchTaskGenerationConfig.model_fields["entity_model_id"].default,
    seed: Annotated[int, typer.Option()] = DeepSearchTaskGenerationConfig.model_fields[
        "seed"
    ].default,
    sample_size: Annotated[
        int, typer.Option(min=1)
    ] = DeepSearchTaskGenerationConfig.model_fields["sample_size"].default,
    num_chunks_per_entity: Annotated[
        int,
        typer.Option(
            min=1,
            help=(
                "Maximum fused chunks scored by the CrossEncoder per entity; "
                "the relevance threshold may return fewer."
            ),
        ),
    ] = DeepSearchTaskGenerationConfig.model_fields["num_chunks_per_entity"].default,
    fact_extraction_chunks_per_request: Annotated[
        int, typer.Option(min=1)
    ] = DeepSearchTaskGenerationConfig.model_fields[
        "fact_extraction_chunks_per_request"
    ].default,
    llm_concurrency: Annotated[
        int, typer.Option(min=1)
    ] = DeepSearchTaskGenerationConfig.model_fields["llm_concurrency"].default,
    retriever_worker_port: Annotated[
        int,
        typer.Option(min=1, max=65535),
    ] = DeepSearchTaskGenerationConfig.model_fields["retriever_worker_port"].default,
) -> None:
    """Discover entities, write fact requests, and upload the input dataset."""
    load_dotenv()
    config = DeepSearchTaskGenerationConfig(
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
    metadata = asyncio.run(prepare_deep_search_tasks_flow(config))
    typer.echo(json.dumps(metadata, indent=2, ensure_ascii=False))


@app.command("generate-qas")
def generate_deep_search_qas(
    prepare_run_directory: Annotated[
        Path,
        typer.Option(
            "--prepare-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help=PREPARE_DIRECTORY_HELP,
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
    ] = QAGenerationConfig.model_fields["qa_pairs_per_entity"].default,
    qa_model_id: Annotated[
        str,
        typer.Option(
            "--qa-model-id",
            help="Model used to generate QA candidates.",
        ),
    ] = QAGenerationConfig.model_fields["qa_model_id"].default,
    complex_pair_ratio: Annotated[
        float,
        typer.Option(
            "--complex-pair-ratio",
            min=0.0,
            max=1.0,
            help="Minimum share of generated QA pairs targeted as complex.",
        ),
    ] = QAGenerationConfig.model_fields["complex_pair_ratio"].default,
    max_qa_generation_attempts: Annotated[
        int,
        typer.Option(
            "--max-qa-generation-attempts",
            min=1,
            help="Maximum QA generation rounds per entity.",
        ),
    ] = QAGenerationConfig.model_fields["max_qa_generation_attempts"].default,
    llm_concurrency: Annotated[
        int,
        typer.Option(
            "--llm-concurrency",
            min=1,
            help="Maximum concurrent QA-generation LLM calls.",
        ),
    ] = QAGenerationConfig.model_fields["llm_concurrency"].default,
    download_timeout: Annotated[
        float,
        typer.Option(
            "--download-timeout",
            min=0.001,
            help="Fireworks batch-output download timeout in seconds.",
        ),
    ] = QAGenerationConfig.model_fields["download_timeout"].default,
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


@app.command("generate-rubrics")
def generate_deep_search_rubrics(
    prepare_run_directory: Annotated[
        Path,
        typer.Option(
            "--prepare-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help=PREPARE_DIRECTORY_HELP,
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="PI model used to generate question-rubric records.",
        ),
    ] = RubricGenerationConfig.model_fields["model"].default,
    solver_model: Annotated[
        str,
        typer.Option(
            "--solver-model",
            help=(
                "Model used for each single-rollout difficulty evaluation. "
                "Other evaluation settings come from the deep-search environment."
            ),
        ),
    ] = RubricGenerationConfig.model_fields["solver_model"].default,
    thinking: Annotated[
        PiThinkingLevel | None,
        typer.Option(
            "--thinking",
            help="PI reasoning level; omit to use the model's default.",
        ),
    ] = RubricGenerationConfig.model_fields["thinking"].default,
    num_question_rubrics: Annotated[
        int,
        typer.Option(
            "--num-rubrics",
            min=0,
            help="Total number of question-rubric records to request.",
        ),
    ] = RubricGenerationConfig.model_fields["num_question_rubrics"].default,
    pi_concurrency: Annotated[
        int,
        typer.Option(
            "--pi-concurrency",
            min=1,
            help="Maximum concurrent PI subprocesses.",
        ),
    ] = RubricGenerationConfig.model_fields["pi_concurrency"].default,
    max_attempts: Annotated[
        int,
        typer.Option(
            "--max-attempts",
            min=1,
            help="Maximum PI attempts for each assigned output slot.",
        ),
    ] = RubricGenerationConfig.model_fields["max_attempts"].default,
    random_entities: Annotated[
        bool,
        typer.Option(
            "--random-entities",
            help="Select entities randomly without replacement.",
        ),
    ] = RubricGenerationConfig.model_fields["random_entities"].default,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Seed used for random entity selection.",
        ),
    ] = RubricGenerationConfig.model_fields["seed"].default,
) -> None:
    """Generate question-rubric records with PI from extracted entity facts."""
    load_dotenv()
    metadata = asyncio.run(
        generate_deep_search_rubrics_flow(
            prepare_run_directory=prepare_run_directory,
            model=model,
            solver_model=solver_model,
            thinking=thinking.value if thinking is not None else None,
            num_question_rubrics=num_question_rubrics,
            pi_concurrency=pi_concurrency,
            max_attempts=max_attempts,
            random_entities=random_entities,
            seed=seed,
        )
    )
    typer.echo(json.dumps(metadata, indent=2, ensure_ascii=False))
