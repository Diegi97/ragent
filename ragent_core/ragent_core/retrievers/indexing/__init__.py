"""Build immutable corpus namespaces and publish their catalog entry last."""

import asyncio
from collections import deque
from collections.abc import Iterable
from typing import Any

from tqdm import tqdm

from ragent_core.retrievers.catalog import (
    CATALOG_SCHEMA,
    CATALOG_SCHEMA_VERSION,
    CorpusCatalogEntry,
    catalog_namespace,
    corpus_namespaces,
)
from ragent_core.retrievers.document import Document, DocumentLike
from ragent_core.retrievers.embedding import EmbeddingBackend
from ragent_core.retrievers.retriever import TurbopufferRetriever
from ragent_core.retrievers.settings import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_LOGICAL_NAMESPACE,
)
from ragent_core.retrievers.storage import (
    CHUNK_SCHEMA,
    DOCUMENT_SCHEMA,
    chunk_row,
    document_row,
)
from ragent_core.retrievers.transport import (
    create_async_turbopuffer_client,
    create_turbopuffer_client,
    turbopuffer_errors,
)

MAX_CONCURRENT_WRITES = 4
DEFAULT_INDEX_BATCH_SIZE = 64


class TurbopufferIndexBuilder:
    def __init__(
        self,
        retriever: TurbopufferRetriever,
        write_client: Any,
        *,
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        show_progress_bar: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Index batch_size must be positive")
        self.retriever = retriever
        self.write_client = write_client
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

    async def publish(
        self,
        table_name: str,
        chunks: Iterable[DocumentLike],
        documents: Iterable[DocumentLike],
    ) -> None:
        with turbopuffer_errors():
            chunks_name, documents_name = corpus_namespaces(
                table_name,
                self.retriever.logical_namespace,
                self.retriever.namespace_prefix,
            )
            document_count = await self._upload_documents(
                self.write_client.namespace(documents_name), documents
            )
            chunk_count, dimensions = await self._upload_chunks(
                self.write_client.namespace(chunks_name), chunks
            )
            if not document_count or not chunk_count:
                raise ValueError(
                    "A ready corpus requires non-empty documents and chunks"
                )
            embedding = self.retriever.embedding
            entry = CorpusCatalogEntry(
                table_name=table_name,
                logical_namespace=self.retriever.logical_namespace,
                chunks_namespace=chunks_name,
                documents_namespace=documents_name,
                schema_version=CATALOG_SCHEMA_VERSION,
                chunk_count=chunk_count,
                document_count=document_count,
                vector_available=embedding is not None,
                vector_dimensions=dimensions,
                embedding_model=embedding.model_name if embedding else "",
                ready=True,
            )
            await self.write_client.namespace(
                catalog_namespace(
                    entry.logical_namespace, self.retriever.namespace_prefix
                )
            ).write(upsert_rows=[entry.to_row()], schema=CATALOG_SCHEMA)

    def require_new_corpus(self, table_name: str) -> None:
        with turbopuffer_errors():
            chunks_name, documents_name = corpus_namespaces(
                table_name,
                self.retriever.logical_namespace,
                self.retriever.namespace_prefix,
            )
            if self.retriever.find_catalog_entry(table_name) is not None or any(
                self.retriever.client.namespace(name).exists()
                for name in (chunks_name, documents_name)
            ):
                raise FileExistsError(
                    f"Turbopuffer corpus {self.retriever.logical_namespace}/{table_name} already exists"
                )

    async def _upload_documents(
        self, namespace: Any, documents: Iterable[DocumentLike]
    ) -> int:
        count = 0
        batch: list[dict[str, Any]] = []
        async with _NamespaceWrites(namespace) as writes:
            for count, raw in enumerate(
                tqdm(
                    documents,
                    desc="Uploading documents",
                    disable=not self.show_progress_bar,
                ),
                start=1,
            ):
                document = raw if isinstance(raw, Document) else Document.from_dict(raw)
                batch.append(document_row(document, count - 1))
                if len(batch) == self.batch_size:
                    await writes.submit(upsert_rows=batch, schema=DOCUMENT_SCHEMA)
                    batch = []
            if batch:
                await writes.submit(upsert_rows=batch, schema=DOCUMENT_SCHEMA)
        return count

    async def _upload_chunks(
        self, namespace: Any, chunks: Iterable[DocumentLike]
    ) -> tuple[int, int]:
        count, dimensions = 0, 0
        batch: list[Document] = []
        async with _NamespaceWrites(namespace) as writes:
            for raw in tqdm(
                chunks, desc="Uploading chunks", disable=not self.show_progress_bar
            ):
                batch.append(
                    raw if isinstance(raw, Document) else Document.from_dict(raw)
                )
                if len(batch) == self.batch_size:
                    dimensions = await self._write_chunks(
                        writes, batch, count, dimensions
                    )
                    count += len(batch)
                    batch = []
            if batch:
                dimensions = await self._write_chunks(writes, batch, count, dimensions)
                count += len(batch)
        return count, dimensions

    async def _write_chunks(
        self,
        writes: "_NamespaceWrites",
        batch: list[Document],
        offset: int,
        dimensions: int,
    ) -> int:
        rows = [
            chunk_row(document, offset + index) for index, document in enumerate(batch)
        ]
        embedding = self.retriever.embedding
        if embedding is None:
            await writes.submit(upsert_rows=rows, schema=CHUNK_SCHEMA)
            return 0
        vectors = await asyncio.to_thread(
            embedding.encode_documents,
            [document.content for document in batch],
            self.batch_size,
        )
        if dimensions and vectors.shape[1] != dimensions:
            raise ValueError("Embedding dimensions changed between index batches")
        for row, vector in zip(rows, vectors, strict=True):
            row["vector"] = vector.tolist()
        await writes.submit(
            upsert_rows=rows, schema=CHUNK_SCHEMA, distance_metric="cosine_distance"
        )
        return vectors.shape[1]


async def build_turbopuffer_index(
    chunks: Iterable[DocumentLike],
    documents: Iterable[DocumentLike],
    table_name: str,
    namespace: str = DEFAULT_LOGICAL_NAMESPACE,
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    device: str | None = None,
    trust_remote_code: bool = True,
    batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
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
    builder = TurbopufferIndexBuilder(
        retriever,
        write_client,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    await asyncio.to_thread(builder.require_new_corpus, table_name)
    if build_embeddings:
        retriever.embedding = await asyncio.to_thread(
            EmbeddingBackend.load,
            model_name,
            device,
            trust_remote_code,
            max_seq_length,
            embedding_service_url,
        )
    if write_client is None:
        async with create_async_turbopuffer_client() as async_client:
            builder.write_client = async_client
            await builder.publish(table_name, chunks, documents)
    else:
        await builder.publish(table_name, chunks, documents)
    return retriever


class _NamespaceWrites:
    """Bound outstanding writes and always observe or cancel every task."""

    def __init__(self, namespace: Any) -> None:
        self.namespace = namespace
        self.pending: deque[asyncio.Task[Any]] = deque()

    async def __aenter__(self) -> "_NamespaceWrites":
        return self

    async def submit(self, **payload: Any) -> None:
        self.pending.append(asyncio.create_task(self.namespace.write(**payload)))
        if len(self.pending) == MAX_CONCURRENT_WRITES:
            await self.pending.popleft()

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc_type is None:
                await asyncio.gather(*self.pending)
        finally:
            for task in self.pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self.pending, return_exceptions=True)
