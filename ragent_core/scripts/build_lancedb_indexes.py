import argparse
import logging
from collections.abc import Iterable

from datasets import Dataset, DatasetDict, load_dataset

from ragent_core.config import HF_TOKEN
from ragent_core.retrievers import MIN_CHUNKS_FOR_ANN_INDEX, Document, LanceDBRetriever
from ragent_core.retrievers.chunking import chunk_documents

logger = logging.getLogger(__name__)

DATASET_ID = "diegi97/ragent_data_sources"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LanceDB indexes for ragent data sources."
    )
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--min-chunks-for-ann-index",
        type=int,
        default=MIN_CHUNKS_FOR_ANN_INDEX,
        help="Build the ANN vector index only when the chunk count is at least this value.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Build only the lexical FTS and scalar indexes, without embeddings.",
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
                f"Data source '{data_source}' not found in {DATASET_ID}. "
                f"Available sources: {available}"
            )
        yield data_source, dataset[data_source]
        return

    for source_name, split in dataset.items():
        yield source_name, split


def build_indexes(args: argparse.Namespace) -> None:

    dataset = load_dataset(DATASET_ID, token=HF_TOKEN)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict from {DATASET_ID}, got {type(dataset)}")

    for source_name, split in _iter_sources(dataset, args.data_source):
        logger.info(
            "Building LanceDB table '%s' in namespace '%s' from %d documents",
            source_name,
            args.namespace,
            len(split),
        )
        documents = Document.from_hf_dataset(split)
        chunks = chunk_documents(documents, chunk_size_tokens=args.chunk_size)
        LanceDBRetriever.build_index(
            chunks=chunks,
            documents=documents,
            table_name=source_name,
            namespace=args.namespace,
            batch_size=args.batch_size,
            device=args.device,
            max_seq_length=args.chunk_size,
            min_chunks_for_ann_index=args.min_chunks_for_ann_index,
            build_embeddings=not args.no_embedding,
        )
        logger.info(
            "Finished LanceDB table '%s' with %d chunks",
            source_name,
            len(chunks),
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    build_indexes(_parse_args())


if __name__ == "__main__":
    main()
