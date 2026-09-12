import argparse
import os

from ragent_core.retrievers.indexing import DEFAULT_INDEX_BATCH_SIZE
from ragent_core.retrievers.settings import (
    DEFAULT_LOGICAL_NAMESPACE,
    DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX,
    EMBEDDING_SERVICE_URL_ENV,
    NAMESPACE_PREFIX_ENV,
)


def add_indexing_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--namespace", default=DEFAULT_LOGICAL_NAMESPACE)
    parser.add_argument(
        "--namespace-prefix",
        default=os.getenv(NAMESPACE_PREFIX_ENV, DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INDEX_BATCH_SIZE)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument(
        "--embedding-service-url", default=os.getenv(EMBEDDING_SERVICE_URL_ENV)
    )
