import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from data_pipelines.pipelines.search_query_generation.config import (
    RetrievalQueriesConfig,
)
from data_pipelines.pipelines.search_query_generation.pipeline import (
    search_query_generation_batch_flow,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Generate search-query-generation JSONL data with Prefect and Phoenix.",
)


@app.callback()
def main() -> None:
    """Search query generation commands."""


@app.command()
def run(
    table_name: Annotated[str, typer.Option(help="Turbopuffer catalog table name.")],
    output_path: Annotated[
        Path,
        typer.Option(help="Base canonical queries.jsonl path."),
    ] = RetrievalQueriesConfig.model_fields["output_path"].default,
    num_queries: Annotated[
        int,
        typer.Option(min=0, help="Number of unique query objects."),
    ] = RetrievalQueriesConfig.model_fields["num_queries"].default,
    hard_negatives_per_query: Annotated[
        int,
        typer.Option(min=0),
    ] = RetrievalQueriesConfig.model_fields["hard_negatives_per_query"].default,
    round_trip_top_k: Annotated[
        int, typer.Option(min=1)
    ] = RetrievalQueriesConfig.model_fields["round_trip_top_k"].default,
    candidate_mining_top_k: Annotated[
        int, typer.Option(min=1)
    ] = RetrievalQueriesConfig.model_fields["candidate_mining_top_k"].default,
    contrastive_candidate_count: Annotated[
        int, typer.Option(min=0)
    ] = RetrievalQueriesConfig.model_fields["contrastive_candidate_count"].default,
    generator_model: Annotated[
        str, typer.Option()
    ] = RetrievalQueriesConfig.model_fields["generator_model"].default,
    logical_namespace: Annotated[
        str,
        typer.Option(help="Turbopuffer logical namespace."),
    ] = RetrievalQueriesConfig.model_fields["logical_namespace"].default,
    seed: Annotated[int, typer.Option()] = RetrievalQueriesConfig.model_fields[
        "seed"
    ].default,
    llm_concurrency: Annotated[
        int, typer.Option(min=1)
    ] = RetrievalQueriesConfig.model_fields["llm_concurrency"].default,
    retriever_concurrency: Annotated[
        int, typer.Option(min=1)
    ] = RetrievalQueriesConfig.model_fields["retriever_concurrency"].default,
) -> None:
    """Run a batch immediately in the current process."""
    load_dotenv()
    config = RetrievalQueriesConfig(
        table_name=table_name,
        output_path=output_path,
        num_queries=num_queries,
        hard_negatives_per_query=hard_negatives_per_query,
        round_trip_top_k=round_trip_top_k,
        candidate_mining_top_k=candidate_mining_top_k,
        contrastive_candidate_count=contrastive_candidate_count,
        generator_model=generator_model,
        logical_namespace=logical_namespace,
        seed=seed,
        llm_concurrency=llm_concurrency,
        retriever_concurrency=retriever_concurrency,
    )
    metadata = asyncio.run(search_query_generation_batch_flow(config))
    typer.echo(json.dumps(metadata, indent=2, ensure_ascii=False))
