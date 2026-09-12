import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from pydantic import ValidationError

from data_pipelines.pipelines.retrieval_evaluation.config import (
    RetrievalEvaluationConfig,
    SearchType,
)
from data_pipelines.pipelines.retrieval_evaluation.evaluator import (
    evaluate_retrieval,
)
from ragent_core.retrievers.settings import (
    EMBEDDING_SERVICE_URL_ENV,
    RERANKER_SERVICE_URL_ENV,
    model_service_url,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Evaluate generated retrieval queries against a Turbopuffer retriever.",
)


@app.callback()
def evaluation_main() -> None:
    """Retrieval evaluation commands."""


def _parse_cutoffs(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter(
            "Cutoffs must be a comma-separated list of integers."
        ) from exc


@app.command("run")
def run_evaluation(
    input_directory: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Generated run directory containing query records and metadata.json.",
        ),
    ],
    search_type: Annotated[
        SearchType,
        typer.Option(help="Turbopuffer retrieval method."),
    ] = RetrievalEvaluationConfig.model_fields["search_type"].default,
    top_k: Annotated[int, typer.Option(min=1)] = RetrievalEvaluationConfig.model_fields[
        "top_k"
    ].default,
    cutoffs: Annotated[
        str,
        typer.Option(help="Strictly increasing metric cutoffs."),
    ] = ",".join(map(str, RetrievalEvaluationConfig.model_fields["cutoffs"].default)),
    embedding_model: Annotated[
        str, typer.Option()
    ] = RetrievalEvaluationConfig.model_fields["embedding_model"].default,
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
    device: Annotated[
        str | None, typer.Option()
    ] = RetrievalEvaluationConfig.model_fields["device"].default,
    max_seq_length: Annotated[
        int | None, typer.Option(min=1)
    ] = RetrievalEvaluationConfig.model_fields["max_seq_length"].default,
    trust_remote_code: Annotated[
        bool,
        typer.Option("--trust-remote-code/--no-trust-remote-code"),
    ] = RetrievalEvaluationConfig.model_fields["trust_remote_code"].default,
    reranker: Annotated[
        bool,
        typer.Option("--reranker/--no-reranker"),
    ] = RetrievalEvaluationConfig.model_fields["reranker"].default,
    reranker_model: Annotated[
        str, typer.Option()
    ] = RetrievalEvaluationConfig.model_fields["reranker_model"].default,
    reranker_service_url: Annotated[
        str | None,
        typer.Option(
            help=(
                "Remote reranker service URL. Defaults to "
                "RAGENT_RERANKER_SERVICE_URL when reranking is enabled."
            )
        ),
    ] = None,
    reranker_candidate_k: Annotated[
        int, typer.Option(min=1)
    ] = RetrievalEvaluationConfig.model_fields["reranker_candidate_k"].default,
    reranker_threshold: Annotated[
        float, typer.Option()
    ] = RetrievalEvaluationConfig.model_fields["reranker_threshold"].default,
    reranker_batch_size: Annotated[
        int, typer.Option(min=1)
    ] = RetrievalEvaluationConfig.model_fields["reranker_batch_size"].default,
) -> None:
    """Run one retrieval configuration against all valid generated queries."""
    load_dotenv()
    embedding_url = embedding_service_url or model_service_url(
        EMBEDDING_SERVICE_URL_ENV
    )
    reranker_url = reranker_service_url or model_service_url(RERANKER_SERVICE_URL_ENV)
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
