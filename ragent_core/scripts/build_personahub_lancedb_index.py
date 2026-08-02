import argparse
import logging
import os

from datasets import Dataset, load_dataset

from ragent_core.retrievers import MIN_CHUNKS_FOR_ANN_INDEX, Document, LanceDBRetriever

logger = logging.getLogger(__name__)

DATASET_ID = "proj-persona/PersonaHub"
DATASET_CONFIG = "persona"
SPLIT = "train"
DEFAULT_TABLE_NAME = "personahub"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standalone LanceDB index for PersonaHub personas."
    )
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally index only the first N personas for smoke tests.",
    )
    parser.add_argument(
        "--min-chunks-for-ann-index",
        type=int,
        default=MIN_CHUNKS_FOR_ANN_INDEX,
        help="Build the ANN vector index only when the row count is at least this value.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument(
        "--embedding-service-url",
        default=os.getenv("RAGENT_EMBEDDING_SERVICE_URL"),
        help="Optional URL of the remote Harrier embedding service.",
    )
    return parser.parse_args()


def _load_personahub(limit: int | None = None) -> list[Document]:
    logger.info("Loading %s/%s split=%s", DATASET_ID, DATASET_CONFIG, SPLIT)
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=SPLIT)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset)}")

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when provided")
        dataset = dataset.select(range(min(limit, len(dataset))))

    documents: list[Document] = []
    for index, row in enumerate(dataset):
        persona = row.get("persona") or ""
        documents.append(
            Document(
                id=index,
                title=f"Persona {index}",
                content=persona,
                metadata={"dataset": DATASET_ID, "config": DATASET_CONFIG},
                document_id=index,
            )
        )
    return documents


def build_index(args: argparse.Namespace) -> None:
    documents = _load_personahub(args.limit)
    logger.info(
        "Building LanceDB PersonaHub table '%s' in namespace '%s' from %d personas",
        args.table_name,
        args.namespace,
        len(documents),
    )

    # PersonaHub records are already short, so each persona is indexed directly
    # as one retrievable chunk and one source document.
    LanceDBRetriever.build_index(
        chunks=documents,
        documents=documents,
        table_name=args.table_name,
        namespace=args.namespace,
        batch_size=args.batch_size,
        device=args.device,
        min_chunks_for_ann_index=args.min_chunks_for_ann_index,
        embedding_service_url=args.embedding_service_url,
    )
    logger.info("Finished LanceDB PersonaHub table '%s'", args.table_name)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    build_index(_parse_args())


if __name__ == "__main__":
    main()
