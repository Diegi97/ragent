from pathlib import Path
from typing import Sequence

from data_pipelines.artifacts.io import write_json
from data_pipelines.artifacts.paths import output_slug
from data_pipelines.pipelines.deep_search_task_generation.prepare.config import (
    DeepSearchTaskGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.models import (
    PreparePaths,
)
from data_pipelines.timestamps import utc_timestamp
from ragent_core.retrievers.document import RetrievalResult


def initialize_prepare_output(
    config: DeepSearchTaskGenerationConfig,
    run_id: str,
) -> PreparePaths:
    directory = config.output_root / (
        f"{output_slug(config.data_source, fallback='source')}_{utc_timestamp()}_"
        f"{config.num_entities}e_{run_id[:8]}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    paths = PreparePaths.in_directory(directory)
    paths.retrieval_debug_directory.mkdir()
    for path in (paths.entities, paths.fact_requests, paths.failures):
        path.touch(exist_ok=False)
    return paths


def write_retrieval_debug(
    path: Path,
    entity_name: str,
    top_k: int,
    chunks: Sequence[RetrievalResult],
) -> None:
    """Write the ungrouped, reranker-filtered chunks for one entity."""
    write_json(
        path,
        {
            "entity": entity_name,
            "top_k": top_k,
            "final_chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.source_document_id,
                    "title": chunk.title,
                    "cross_encoder_score": chunk.score,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
                for chunk in sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
            ],
        },
    )
