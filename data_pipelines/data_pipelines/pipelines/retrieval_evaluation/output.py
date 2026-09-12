from pathlib import Path

from data_pipelines.pipelines.retrieval_evaluation.config import (
    RetrievalEvaluationConfig,
)
from data_pipelines.timestamps import TimestampPrecision, utc_timestamp


def create_output_directory(config: RetrievalEvaluationConfig) -> Path:
    timestamp = utc_timestamp(TimestampPrecision.MICROSECONDS)
    reranker_suffix = "_reranked" if config.reranker else ""
    output_directory = config.input_directory / (
        f"{timestamp}_retrieval-evaluation_{config.search_type.value}{reranker_suffix}"
    )
    output_directory.mkdir(exist_ok=False)
    return output_directory
