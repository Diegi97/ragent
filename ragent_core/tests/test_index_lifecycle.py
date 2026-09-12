import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from ragent_core.retrievers.catalog import catalog_namespace, corpus_namespaces
from ragent_core.retrievers.document import DOCUMENT_ID_KEY, Document
from ragent_core.retrievers.embedding import EmbeddingBackend
from ragent_core.retrievers.indexing import build_turbopuffer_index
from ragent_core.retrievers.mode import RetrievalMode
from ragent_core.retrievers.settings import DEFAULT_EMBEDDING_MODEL_NAME


class Namespace:
    def __init__(self):
        self.rows = []
        self.writes = []
        self.fail = False

    def exists(self):
        return bool(self.rows)

    async def write(self, **payload):
        if self.fail:
            raise RuntimeError("write failed")
        self.writes.append(payload)
        self.rows.extend(payload["upsert_rows"])

    def query(self, **request):
        rows = self.rows
        filters = request.get("filters")
        if filters and filters[1] == "Eq":
            rows = [row for row in rows if row[filters[0]] == filters[2]]
        return SimpleNamespace(
            rows=[
                SimpleNamespace(model_dump=lambda by_alias, row=row: row)
                for row in rows[: request["top_k"]]
            ]
        )


class Client:
    def __init__(self):
        self.namespaces = {}

    def namespace(self, name):
        return self.namespaces.setdefault(name, Namespace())


def test_lexical_index_publication_to_runtime_round_trip():
    client = Client()
    document = Document(id=0, title="Title", content="Full text")
    chunk = Document(id=1, title="Title", content="Full", document_id=0)
    retriever = asyncio.run(
        build_turbopuffer_index(
            [chunk],
            [document],
            "corpus",
            namespace="test",
            client=client,
            write_client=client,
            build_embeddings=False,
            show_progress_bar=False,
        )
    )
    assert retriever.get_document(0, "corpus") == document
    assert (
        retriever.retrieve("Full", "corpus", retrieval_mode=RetrievalMode.BM25)[
            0
        ].metadata[DOCUMENT_ID_KEY]
        == 0
    )
    with pytest.raises(ValueError, match="lexical-only"):
        retriever.retrieve("Full", "corpus", retrieval_mode=RetrievalMode.DENSE)
    with pytest.raises(FileExistsError):
        asyncio.run(
            build_turbopuffer_index(
                [chunk],
                [document],
                "corpus",
                namespace="test",
                client=client,
                write_client=client,
                build_embeddings=False,
            )
        )


@pytest.mark.parametrize("fail_write", [False, True])
def test_failed_or_empty_upload_never_publishes_ready_catalog(fail_write):
    client = Client()
    if fail_write:
        _, documents_name = corpus_namespaces("corpus", "test")
        client.namespace(documents_name).fail = True
    documents = [Document(id=0, content="text")] if fail_write else []
    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(
            build_turbopuffer_index(
                [],
                documents,
                "corpus",
                namespace="test",
                client=client,
                write_client=client,
                build_embeddings=False,
                show_progress_bar=False,
            )
        )
    assert not client.namespace(catalog_namespace("test")).exists()


def test_vector_build_offloads_encoder_and_records_dimensions(monkeypatch):

    owner_thread = threading.get_ident()
    encoder_threads = []

    class Model:
        def encode(self, texts, **kwargs):
            encoder_threads.append(threading.get_ident())
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(
        EmbeddingBackend,
        "load",
        lambda *args: EmbeddingBackend(DEFAULT_EMBEDDING_MODEL_NAME, model=Model()),
    )
    client = Client()
    retriever = asyncio.run(
        build_turbopuffer_index(
            [Document(id=1, content="chunk", document_id=0)],
            [Document(id=0, content="full")],
            "vectors",
            namespace="test",
            client=client,
            write_client=client,
            show_progress_bar=False,
        )
    )
    entry = retriever.find_catalog_entry("vectors")
    assert entry.vector_dimensions == 2
    assert entry.vector_available
    assert encoder_threads and all(thread != owner_thread for thread in encoder_threads)
    result = retriever.retrieve("query", "vectors", retrieval_mode=RetrievalMode.DENSE)
    assert result[0].metadata[DOCUMENT_ID_KEY] == 0


def test_failed_write_cancels_other_pending_namespace_writes():

    client = Client()
    stopped = []

    class FailingNamespace(Namespace):
        async def write(self, **payload):
            if payload["upsert_rows"][0]["id"] == 0:
                raise RuntimeError("first write failed")
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(True)

    _, documents_name = corpus_namespaces("corpus", "test")
    client.namespaces[documents_name] = FailingNamespace()
    with pytest.raises(RuntimeError, match="first write failed"):
        asyncio.run(
            build_turbopuffer_index(
                [],
                [Document(id=i, content="text") for i in range(5)],
                "corpus",
                namespace="test",
                client=client,
                write_client=client,
                batch_size=1,
                build_embeddings=False,
                show_progress_bar=False,
            )
        )
    assert stopped
    assert not client.namespace(catalog_namespace("test")).exists()
