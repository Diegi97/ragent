import argparse
import asyncio
import logging

from datasets import Dataset, load_dataset

from ragent_core.retrievers.document import Document
from ragent_core.retrievers.indexing import (
    build_turbopuffer_index,
)
from ragent_core.retrievers.indexing.options import add_indexing_options

logger = logging.getLogger(__name__)

DATASET_ID = "proj-persona/PersonaHub"
DATASET_CONFIG = "persona"
SPLIT = "train"
DEFAULT_TABLE_NAME = "personahub"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standalone Turbopuffer corpus for PersonaHub."
    )
    add_indexing_options(parser)
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally index only the first N personas for smoke tests.",
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

    return [
        Document(
            id=index,
            title=f"Persona {index}",
            content=str(row.get("persona") or ""),
            metadata={"dataset": DATASET_ID, "config": DATASET_CONFIG},
            document_id=index,
        )
        for index, row in enumerate(dataset)
    ]


async def build_index(args: argparse.Namespace) -> None:
    documents = await asyncio.to_thread(_load_personahub, args.limit)
    logger.info(
        "Building Turbopuffer PersonaHub corpus '%s' in logical namespace '%s' "
        "from %d personas",
        args.table_name,
        args.namespace,
        len(documents),
    )
    await build_turbopuffer_index(
        chunks=documents,
        documents=documents,
        table_name=args.table_name,
        namespace=args.namespace,
        namespace_prefix=args.namespace_prefix,
        batch_size=args.batch_size,
        device=args.device,
        embedding_service_url=args.embedding_service_url,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(build_index(_parse_args()))


if __name__ == "__main__":
    main()
