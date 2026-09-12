import argparse
import asyncio
import logging
from collections.abc import Iterable

from datasets import Dataset, DatasetDict, load_dataset

from ragent_core.config import HF_TOKEN
from ragent_core.data_sources.records import DATA_SOURCES_DATASET_ID
from ragent_core.retrievers.chunking import chunk_documents
from ragent_core.retrievers.document import Document
from ragent_core.retrievers.indexing import (
    build_turbopuffer_index,
)
from ragent_core.retrievers.indexing.options import add_indexing_options

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Turbopuffer indexes for ragent data sources."
    )
    add_indexing_options(parser)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--data-source", default=None)
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Build a lexical-only corpus without vectors.",
    )
    return parser.parse_args()


def _iter_sources(
    dataset: DatasetDict,
    data_source: str | None,
) -> Iterable[tuple[str, Dataset]]:
    if data_source is not None:
        if data_source not in dataset:
            available = ", ".join(sorted(dataset.keys()))
            raise KeyError(
                f"Data source '{data_source}' not found in {DATA_SOURCES_DATASET_ID}. "
                f"Available sources: {available}"
            )
        yield data_source, dataset[data_source]
        return
    yield from dataset.items()


async def build_indexes(args: argparse.Namespace) -> None:
    dataset = await asyncio.to_thread(
        load_dataset, DATA_SOURCES_DATASET_ID, token=HF_TOKEN
    )
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict from {DATA_SOURCES_DATASET_ID}, got {type(dataset)}"
        )

    for source_name, split in _iter_sources(dataset, args.data_source):
        logger.info(
            "Building Turbopuffer corpus '%s' in logical namespace '%s' from %d "
            "documents",
            source_name,
            args.namespace,
            len(split),
        )
        documents = await asyncio.to_thread(Document.from_hf_dataset, split)
        chunks = await asyncio.to_thread(
            chunk_documents, documents, chunk_size_tokens=args.chunk_size
        )
        await build_turbopuffer_index(
            chunks=chunks,
            documents=documents,
            table_name=source_name,
            namespace=args.namespace,
            namespace_prefix=args.namespace_prefix,
            batch_size=args.batch_size,
            device=args.device,
            max_seq_length=args.chunk_size,
            embedding_service_url=args.embedding_service_url,
            build_embeddings=not args.no_embedding,
        )
        logger.info(
            "Finished Turbopuffer corpus '%s' with %d chunks",
            source_name,
            len(chunks),
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(build_indexes(_parse_args()))
