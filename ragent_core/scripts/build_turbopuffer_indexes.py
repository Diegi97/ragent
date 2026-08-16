import argparse
import asyncio
import json
import logging
import os
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from turbopuffer import AsyncTurbopuffer

from ragent_core.config import HF_TOKEN
from ragent_core.retrievers import (
    Document,
    DocumentLike,
    TurbopufferRetriever,
)
from ragent_core.retrievers.chunking import chunk_documents
from ragent_core.retrievers.retriever import (
    CATALOG_SCHEMA_VERSION,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_LOGICAL_NAMESPACE,
    DEFAULT_TURBOPUFFER_REGION,
    CorpusCatalogEntry,
    catalog_namespace,
    corpus_namespaces,
    create_turbopuffer_client,
)

logger = logging.getLogger(__name__)

DATASET_ID = "diegi97/ragent_data_sources"
MAX_CONCURRENT_WRITES = 4

_CATALOG_SCHEMA = {
    "table_name": {"type": "string", "filterable": True},
    "logical_namespace": {"type": "string", "filterable": True},
    "chunks_namespace": {"type": "string", "filterable": False},
    "documents_namespace": {"type": "string", "filterable": False},
    "schema_version": {"type": "uint", "filterable": True},
    "chunk_count": {"type": "uint", "filterable": False},
    "document_count": {"type": "uint", "filterable": False},
    "vector_available": {"type": "bool", "filterable": True},
    "vector_dimensions": {"type": "uint", "filterable": False},
    "embedding_model": {"type": "string", "filterable": False},
    "ready": {"type": "bool", "filterable": True},
}

_CHUNK_SCHEMA = {
    "title": {"type": "string", "filterable": False},
    "content": {
        "type": "string",
        "filterable": False,
        "regex": True,
        "full_text_search": {
            "tokenizer": "word_v4",
            "case_sensitive": False,
            "stemming": False,
            "remove_stopwords": False,
            "ascii_folding": False,
        },
    },
    "metadata_json": {"type": "string", "filterable": False},
    "document_id": {"type": "uint", "filterable": False},
    "source_index": {"type": "uint", "filterable": True},
}

_DOCUMENT_SCHEMA = {
    "title": {"type": "string", "filterable": False},
    "content": {"type": "string", "filterable": False},
    "metadata_json": {"type": "string", "filterable": False},
    "document_id": {"type": "uint", "filterable": False},
    "source_index": {"type": "uint", "filterable": True},
}


def _document(raw: DocumentLike) -> Document:
    return raw if isinstance(raw, Document) else Document.from_dict(raw)


def _row(
    document: Document,
    source_index: int,
    *,
    include_document_id: bool,
) -> dict[str, Any]:
    return {
        "id": document.id,
        "title": document.title,
        "content": document.content,
        "metadata_json": json.dumps(document.metadata or {}, ensure_ascii=False),
        "document_id": document.document_id if include_document_id else None,
        "source_index": source_index,
    }


def _encode(
    retriever: TurbopufferRetriever,
    documents: Sequence[Document],
    batch_size: int,
) -> np.ndarray:
    texts = [document.content for document in documents]
    if retriever._embedding_client is not None:
        return np.asarray(
            retriever._embedding_client.encode_documents(texts), dtype=np.float32
        )
    with torch.inference_mode():
        return np.asarray(
            retriever._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )


async def _schedule_write(
    writes: deque[asyncio.Task[Any]],
    write: Any,
) -> None:
    writes.append(asyncio.create_task(write))
    await asyncio.sleep(0)
    if len(writes) == MAX_CONCURRENT_WRITES:
        await writes.popleft()


async def _finish_writes(writes: deque[asyncio.Task[Any]]) -> None:
    if writes:
        await asyncio.gather(*writes)


async def _upload_documents(
    namespace: Any,
    documents: Iterable[DocumentLike],
    batch_size: int,
    show_progress_bar: bool,
) -> int:
    batch: list[dict[str, Any]] = []
    count = 0
    writes: deque[asyncio.Task[Any]] = deque()
    for source_index, raw in enumerate(
        tqdm(
            documents,
            desc="Uploading documents",
            disable=not show_progress_bar,
            unit="doc",
        )
    ):
        batch.append(_row(_document(raw), source_index, include_document_id=False))
        count += 1
        if len(batch) == batch_size:
            await _schedule_write(
                writes,
                namespace.write(
                    upsert_rows=batch,
                    schema=_DOCUMENT_SCHEMA,
                ),
            )
            batch = []
    if batch:
        await _schedule_write(
            writes,
            namespace.write(
                upsert_rows=batch,
                schema=_DOCUMENT_SCHEMA,
            ),
        )
    await _finish_writes(writes)
    return count


async def _write_chunks(
    namespace: Any,
    rows: list[dict[str, Any]],
    vectors: np.ndarray | None,
) -> None:
    if vectors is not None:
        for row, vector in zip(rows, vectors):
            row["vector"] = vector.tolist()
        await namespace.write(
            upsert_rows=rows,
            schema=_CHUNK_SCHEMA,
            distance_metric="cosine_distance",
        )
    else:
        await namespace.write(upsert_rows=rows, schema=_CHUNK_SCHEMA)


async def _upload_chunks(
    retriever: TurbopufferRetriever,
    namespace: Any,
    chunks: Iterable[DocumentLike],
    batch_size: int,
    show_progress_bar: bool,
    build_embeddings: bool,
) -> tuple[int, int]:
    pending: list[Document] = []
    count = 0
    vector_dimensions = 0
    writes: deque[asyncio.Task[Any]] = deque()

    async def upload() -> None:
        nonlocal count, vector_dimensions
        rows = [
            _row(document, count + offset, include_document_id=True)
            for offset, document in enumerate(pending)
        ]
        vectors = _encode(retriever, pending, batch_size) if build_embeddings else None
        if vectors is not None:
            vector_dimensions = vectors.shape[1]
        await _schedule_write(writes, _write_chunks(namespace, rows, vectors))
        count += len(pending)
        pending.clear()

    for raw in tqdm(
        chunks,
        desc=(
            "Embedding and uploading chunks" if build_embeddings else "Uploading chunks"
        ),
        disable=not show_progress_bar,
        unit="chunk",
    ):
        pending.append(_document(raw))
        if len(pending) == batch_size:
            await upload()
    if pending:
        await upload()
    await _finish_writes(writes)
    return count, vector_dimensions


async def _publish(
    write_client: Any,
    retriever: TurbopufferRetriever,
    chunks_name: str,
    documents_name: str,
    chunks: Iterable[DocumentLike],
    documents: Iterable[DocumentLike],
    table_name: str,
    namespace: str,
    model_name: str,
    batch_size: int,
    show_progress_bar: bool,
    build_embeddings: bool,
) -> None:
    document_count = await _upload_documents(
        write_client.namespace(documents_name),
        documents,
        batch_size,
        show_progress_bar,
    )
    chunk_count, vector_dimensions = await _upload_chunks(
        retriever,
        write_client.namespace(chunks_name),
        chunks,
        batch_size,
        show_progress_bar,
        build_embeddings,
    )
    entry = CorpusCatalogEntry(
        table_name=table_name,
        logical_namespace=namespace,
        chunks_namespace=chunks_name,
        documents_namespace=documents_name,
        schema_version=CATALOG_SCHEMA_VERSION,
        chunk_count=chunk_count,
        document_count=document_count,
        vector_available=build_embeddings,
        vector_dimensions=vector_dimensions,
        embedding_model=model_name if build_embeddings else "",
        ready=True,
    )
    await write_client.namespace(
        catalog_namespace(namespace, retriever.namespace_prefix)
    ).write(
        upsert_rows=[{"id": table_name, **asdict(entry)}],
        schema=_CATALOG_SCHEMA,
    )


async def build_turbopuffer_index(
    chunks: Iterable[DocumentLike],
    documents: Iterable[DocumentLike],
    table_name: str,
    namespace: str = DEFAULT_LOGICAL_NAMESPACE,
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    device: str | None = None,
    trust_remote_code: bool = True,
    batch_size: int = 64,
    show_progress_bar: bool = True,
    max_seq_length: int | None = None,
    embedding_service_url: str | None = None,
    build_embeddings: bool = True,
    *,
    client: Any = None,
    write_client: Any = None,
    namespace_prefix: str | None = None,
) -> TurbopufferRetriever:
    retriever = TurbopufferRetriever(
        client=client or create_turbopuffer_client(),
        namespace=namespace,
        namespace_prefix=namespace_prefix,
        reranker_model_name=None,
    )
    chunks_name, documents_name = corpus_namespaces(
        table_name, namespace, retriever.namespace_prefix
    )
    if retriever._find_catalog_entry(table_name) is not None:
        raise FileExistsError(
            f"Turbopuffer corpus {namespace}/{table_name} already exists."
        )
    if (
        retriever.client.namespace(chunks_name).exists()
        or retriever.client.namespace(documents_name).exists()
    ):
        raise FileExistsError(
            f"Turbopuffer namespaces for {namespace}/{table_name} already exist."
        )

    if build_embeddings:
        retriever._load_embedding_backend(
            model_name=model_name,
            device=device,
            trust_remote_code=trust_remote_code,
            max_seq_length=max_seq_length,
            embedding_service_url=embedding_service_url,
        )

    if write_client is None:
        async with AsyncTurbopuffer(
            region=os.getenv("TURBOPUFFER_REGION") or DEFAULT_TURBOPUFFER_REGION,
            timeout=60.0,
            max_retries=4,
        ) as async_client:
            await _publish(
                async_client,
                retriever,
                chunks_name,
                documents_name,
                chunks,
                documents,
                table_name,
                namespace,
                model_name,
                batch_size,
                show_progress_bar,
                build_embeddings,
            )
    else:
        await _publish(
            write_client,
            retriever,
            chunks_name,
            documents_name,
            chunks,
            documents,
            table_name,
            namespace,
            model_name,
            batch_size,
            show_progress_bar,
            build_embeddings,
        )
    return retriever


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Turbopuffer indexes for ragent data sources."
    )
    parser.add_argument("--namespace", default="default")
    parser.add_argument(
        "--namespace-prefix",
        default=os.getenv("TURBOPUFFER_NAMESPACE_PREFIX", "ragent"),
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument(
        "--embedding-service-url",
        default=os.getenv("RAGENT_EMBEDDING_SERVICE_URL"),
    )
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
                f"Data source '{data_source}' not found in {DATASET_ID}. "
                f"Available sources: {available}"
            )
        yield data_source, dataset[data_source]
        return
    yield from dataset.items()


async def build_indexes(args: argparse.Namespace) -> None:
    dataset = load_dataset(DATASET_ID, token=HF_TOKEN)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict from {DATASET_ID}, got {type(dataset)}")

    for source_name, split in _iter_sources(dataset, args.data_source):
        logger.info(
            "Building Turbopuffer corpus '%s' in logical namespace '%s' from %d "
            "documents",
            source_name,
            args.namespace,
            len(split),
        )
        documents = Document.from_hf_dataset(split)
        chunks = chunk_documents(documents, chunk_size_tokens=args.chunk_size)
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


if __name__ == "__main__":
    main()
