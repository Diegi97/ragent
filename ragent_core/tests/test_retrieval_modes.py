from dataclasses import replace
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import turbopuffer

from ragent_core.provider_errors import ProviderFailureKind
from ragent_core.retrievers import retriever as implementation
from ragent_core.retrievers.catalog import (
    CATALOG_SCHEMA_VERSION,
    CorpusCatalogEntry,
    catalog_namespace,
)
from ragent_core.retrievers.document import DOCUMENT_ID_KEY, Document
from ragent_core.retrievers.embedding import EmbeddingBackend
from ragent_core.retrievers.mode import RetrievalMode
from ragent_core.retrievers.reranking import CrossEncoderReranker
from ragent_core.retrievers.retriever import TurbopufferRetriever
from ragent_core.retrievers.settings import DEFAULT_EMBEDDING_MODEL_NAME
from ragent_core.retrievers.storage import SOURCE_INDEX_FIELD, chunk_row, document_row
from ragent_core.retrievers.transport import RetrievalServiceError


def response(rows):
    return SimpleNamespace(
        rows=[SimpleNamespace(model_dump=lambda by_alias, row=row: row) for row in rows]
    )


class Namespace:
    def __init__(self, rows):
        self.rows = rows
        self.requests = []

    def exists(self):
        return bool(self.rows)

    def query(self, **kwargs):
        self.requests.append(kwargs)
        rows = self.rows
        filters = kwargs.get("filters")
        if filters:
            if filters[0] == "And":
                cursor = filters[1][1][2]
                rows = [row for row in rows if row[SOURCE_INDEX_FIELD] > cursor]
            elif filters[1] == "Eq":
                rows = [row for row in rows if row[filters[0]] == filters[2]]
        return response(rows[: kwargs["top_k"]])

    def multi_query(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(results=[response(list(reversed(self.rows)))])


def retriever_fixture():
    entry = CorpusCatalogEntry(
        table_name="corpus",
        logical_namespace="test",
        chunks_namespace="physical.chunks",
        documents_namespace="physical.documents",
        schema_version=CATALOG_SCHEMA_VERSION,
        chunk_count=3,
        document_count=1,
        vector_available=True,
        vector_dimensions=2,
        embedding_model=DEFAULT_EMBEDDING_MODEL_NAME,
        ready=True,
    )
    namespaces = {
        catalog_namespace("test"): Namespace([entry.to_row()]),
        entry.documents_namespace: Namespace(
            [document_row(Document(id=0, content="Full"), 0)]
        ),
        entry.chunks_namespace: Namespace(
            [
                chunk_row(Document(id=i, content=f"Chunk {i}", document_id=0), i)
                for i in range(3)
            ]
        ),
    }
    client = SimpleNamespace(namespace=lambda name: namespaces[name])
    retriever = TurbopufferRetriever(
        client=client, namespace="test", reranker_model_name=None
    )
    retriever.embedding = EmbeddingBackend(
        DEFAULT_EMBEDDING_MODEL_NAME,
        model=SimpleNamespace(encode=lambda texts, **kwargs: np.ones((len(texts), 2))),
    )
    return retriever, namespaces, entry


@pytest.mark.parametrize(
    "mode,operation",
    [
        (RetrievalMode.BM25, "BM25"),
        (RetrievalMode.DENSE, "ANN"),
        (RetrievalMode.HYBRID, "RRF"),
    ],
)
def test_modes_use_canonical_sdk_dispatch_and_decode_parent_ids(mode, operation):
    retriever, namespaces, entry = retriever_fixture()
    results = retriever.retrieve("query", "corpus", top_k=2, retrieval_mode=mode)
    request = namespaces[entry.chunks_namespace].requests[-1]
    assert operation in str(request.get("rank_by", request.get("rerank_by")))
    assert all(result.metadata[DOCUMENT_ID_KEY] == 0 for result in results)


def test_hybrid_reranker_filters_logits_before_top_k():
    retriever, _, _ = retriever_fixture()
    ranker = SimpleNamespace(
        rank=lambda query, texts, **kwargs: [
            {"corpus_id": 1, "score": 6.0},
            {"corpus_id": 0, "score": 4.0},
            {"corpus_id": 2, "score": 1.0},
        ]
    )
    retriever.reranker = CrossEncoderReranker(ranker=ranker, rerank_threshold=5)
    results = retriever.retrieve(
        "query", "corpus", top_k=2, retrieval_mode=RetrievalMode.HYBRID_RERANKED
    )
    assert [(result.id, result.score) for result in results] == [(1, 6.0)]


def test_scan_paginates_using_source_index(monkeypatch):

    monkeypatch.setattr(implementation, "SCAN_PAGE_SIZE", 2)
    retriever, namespaces, entry = retriever_fixture()
    assert len(retriever.scan_chunks("corpus", "Chunk")) == 3
    requests = namespaces[entry.chunks_namespace].requests
    assert len(requests) == 2
    assert requests[1]["filters"][1][1] == (SOURCE_INDEX_FIELD, "Gt", 1)


def test_incomplete_catalog_rejects_reads_before_physical_query():
    retriever, namespaces, entry = retriever_fixture()
    namespaces[catalog_namespace("test")].rows = [replace(entry, ready=False).to_row()]
    with pytest.raises(RuntimeError, match="incomplete"):
        retriever.get_document(0, "corpus")
    assert namespaces[entry.documents_namespace].requests == []


def test_scan_rejects_nonadvancing_cursor(monkeypatch):
    monkeypatch.setattr(implementation, "SCAN_PAGE_SIZE", 2)
    retriever, namespaces, entry = retriever_fixture()
    namespace = namespaces[entry.chunks_namespace]
    calls = []

    def repeat_page(**kwargs):
        calls.append(kwargs)
        assert len(calls) <= 2, "scan failed to reject a repeated full page"
        return response(namespace.rows[:2])

    monkeypatch.setattr(namespace, "query", repeat_page)
    with pytest.raises(RuntimeError, match="pagination did not advance"):
        retriever.scan_chunks("corpus", "Chunk")
    assert len(calls) == 2


@pytest.mark.parametrize(
    "status,kind",
    [
        (None, ProviderFailureKind.TRANSPORT),
        (401, ProviderFailureKind.AUTHENTICATION),
        (429, ProviderFailureKind.RATE_LIMIT),
    ],
)
def test_retrieval_sdk_failures_are_normalized(monkeypatch, status, kind):
    retriever, namespaces, entry = retriever_fixture()
    request = httpx.Request("POST", "https://provider.test")
    sdk_error = (
        turbopuffer.APIConnectionError(message="vendor-private-detail", request=request)
        if status is None
        else turbopuffer.APIStatusError(
            "vendor-private-detail",
            response=httpx.Response(status, request=request),
            body={},
        )
    )

    def unavailable(**kwargs):
        raise sdk_error

    monkeypatch.setattr(namespaces[entry.chunks_namespace], "query", unavailable)
    with pytest.raises(RetrievalServiceError) as failure:
        retriever.retrieve_bm25("query", "corpus")
    assert failure.value.kind is kind
    assert failure.value.__cause__ is sdk_error
    assert "vendor-private-detail" not in str(failure.value)
