import json
import xml.etree.ElementTree as ET
from dataclasses import replace

import httpx
import numpy as np
import pytest

from ragent_core.retrievers.agent_retriever import AgentRetriever
from ragent_core.retrievers.catalog import CATALOG_SCHEMA_VERSION, CorpusCatalogEntry
from ragent_core.retrievers.document import DOCUMENT_ID_KEY, Document, RetrievalResult
from ragent_core.retrievers.embedding import EmbeddingBackend
from ragent_core.retrievers.model_contract import (
    EMBEDDING_DOCUMENTS_ROUTE,
    EMBEDDING_QUERY_ROUTE,
    RERANK_ROUTE,
    ModelServiceError,
    normalize_embeddings,
    normalize_rank_results,
)
from ragent_core.retrievers.retriever import TurbopufferRetriever
from ragent_core.retrievers.service_clients.embedding import EmbeddingServiceClient
from ragent_core.retrievers.service_clients.reranker import CrossEncoderServiceClient
from ragent_core.retrievers.settings import DEFAULT_EMBEDDING_MODEL_NAME
from ragent_core.retrievers.storage import (
    chunk_row,
    deserialize_document,
    deserialize_result,
    document_row,
)
from ragent_core.retrievers.text_scan import TextScan


def catalog_entry():
    return CorpusCatalogEntry(
        table_name="corpus",
        logical_namespace="test",
        chunks_namespace="physical.chunks",
        documents_namespace="physical.documents",
        schema_version=CATALOG_SCHEMA_VERSION,
        chunk_count=1,
        document_count=1,
        vector_available=True,
        vector_dimensions=2,
        embedding_model=DEFAULT_EMBEDDING_MODEL_NAME,
        ready=True,
    )


def test_catalog_round_trip_preserves_physical_indirection():
    entry = catalog_entry()
    restored = CorpusCatalogEntry.from_row(entry.to_row())
    restored.require_identity("test", "corpus")
    assert restored == entry
    with pytest.raises(ValueError, match="identity"):
        restored.require_identity("other", "corpus")


@pytest.mark.parametrize(
    "field,value",
    [
        ("ready", "false"),
        ("chunk_count", True),
        ("vector_dimensions", 0),
        ("chunks_namespace", None),
    ],
)
def test_catalog_rejects_invalid_external_fields(field, value):
    row = catalog_entry().to_row()
    row[field] = value
    with pytest.raises(ValueError):
        CorpusCatalogEntry.from_row(row)


def test_vector_model_identity_checked_only_for_vector_search(monkeypatch):
    retriever = TurbopufferRetriever(namespace="test", reranker_model_name=None)
    retriever.embedding = EmbeddingBackend("different-model")
    monkeypatch.setattr(retriever, "_catalog_entry", lambda _: catalog_entry())
    with pytest.raises(ValueError, match="does not match catalog"):
        retriever.retrieve_dense("query", "corpus")


def test_document_and_chunk_storage_round_trip():
    document = Document(id="source", content="Text", metadata={"tag": "a"})
    assert deserialize_document(document_row(document, 0)) == document
    chunk = Document(id=1, content="chunk", document_id=0)
    assert deserialize_document(chunk_row(chunk, 0)) == chunk
    with pytest.raises(ValueError, match="non-negative integer"):
        chunk_row(replace(chunk, document_id="source"), 0)


@pytest.mark.parametrize(DOCUMENT_ID_KEY, [True, None, [], {}, " "])
def test_document_mapping_rejects_invalid_ids(document_id):
    with pytest.raises(ValueError):
        Document.from_dict({"id": document_id})


@pytest.mark.parametrize(
    "payload", [[[]], [[1], [2]], [[float("nan")]], ["x"], [[1], [2, 3]]]
)
def test_embedding_contract_rejects_malformed_outputs(payload):
    with pytest.raises(ModelServiceError):
        normalize_embeddings(payload, 1)


def test_local_embedding_uses_the_same_output_boundary():
    class Model:
        def encode(self, texts, **kwargs):
            return [[]]

    with pytest.raises(ModelServiceError):
        EmbeddingBackend("test", model=Model()).encode_documents(["a"])


@pytest.mark.parametrize(
    "payload",
    [
        [{"corpus_id": True, "score": 1.0}],
        [{"corpus_id": 0.5, "score": 1.0}],
        [{"corpus_id": 0, "score": float("inf")}],
        [],
    ],
)
def test_reranker_rejects_invalid_external_results(payload):
    with pytest.raises(ModelServiceError):
        normalize_rank_results(payload, 1)


def test_embedding_http_routes_and_payloads():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=[[1.0, 2.0]])

    with EmbeddingServiceClient(
        "https://models.test", transport=httpx.MockTransport(respond)
    ) as client:
        assert np.array_equal(client.encode_queries(["q"]), [[1.0, 2.0]])
        client.encode_documents(["d"])
    assert [request.url.path for request in requests] == [
        EMBEDDING_QUERY_ROUTE,
        EMBEDDING_DOCUMENTS_ROUTE,
    ]
    assert requests[0].content == b'{"texts":["q"]}'


def test_http_failure_has_stable_adapter_error_and_cause():
    with EmbeddingServiceClient(
        "https://models.test",
        transport=httpx.MockTransport(lambda _: httpx.Response(401)),
    ) as client:
        with pytest.raises(ModelServiceError, match="HTTP 401") as failure:
            client.encode_queries(["q"])
    assert isinstance(failure.value.__cause__, httpx.HTTPStatusError)


class ToolBackend:
    def retrieve(self, *args, **kwargs):
        return [
            RetrievalResult(
                id=1, title="A & B", content="<evidence>", metadata={DOCUMENT_ID_KEY: 0}
            )
        ]

    def get_document(self, doc_id, table_name):
        if doc_id == 0:
            return Document(id=0, content="<document text>")
        if doc_id == 1:
            return None
        raise RuntimeError("backend unavailable")


def test_tool_xml_quotes_attributes_and_escapes_evidence():
    tools = AgentRetriever(ToolBackend())
    search = ET.fromstring(tools.search_tool(['a "quoted" query'], "corpus"))
    assert search.find("query").attrib["value"] == 'a "quoted" query'
    assert search.find(".//snippet").text == "<evidence>"
    read = ET.fromstring(tools.read_tool([0], "corpus"))
    assert read.find("document").attrib["id"] == "0"


def test_missing_document_and_backend_failure_remain_distinct():
    tools = AgentRetriever(ToolBackend())
    assert "not found" in tools.read_tool([1], "corpus")
    with pytest.raises(RuntimeError, match="backend unavailable"):
        tools.read_tool([2], "corpus")


def test_scan_literal_escaping_and_regex_rejection():
    scan = TextScan.prepare("a.b [x]", True, False)
    assert scan.server_regex == r"(?i)a\.b \[x\]"
    assert "<id>0</id>" in scan.render([("A.B [x]", 0, "Title")], 1, 20)
    with pytest.raises(ValueError, match="lookaround"):
        TextScan.prepare("(?=bad)", False, False)


@pytest.mark.parametrize(
    "parent_id, metadata",
    [
        (None, {}),
        (True, {}),
        ("0", {}),
        (-1, {}),
        (0, {DOCUMENT_ID_KEY: 1}),
        (0, {DOCUMENT_ID_KEY: False}),
    ],
)
def test_chunk_result_rejects_invalid_or_conflicting_parent_identity(
    parent_id, metadata
):
    row = document_row(Document(id=9, content="chunk", metadata=metadata), 0)
    row[DOCUMENT_ID_KEY] = parent_id
    with pytest.raises(ValueError):
        deserialize_result(row)


def test_chunk_result_exposes_valid_parent_zero():
    result = deserialize_result(
        chunk_row(Document(id=9, content="chunk", document_id=0), 0)
    )
    assert result.source_document_id == 0
    assert result.metadata[DOCUMENT_ID_KEY] == 0


def test_remote_reranker_payload_and_top_k():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, json=[{"corpus_id": 1, "score": 2.0}, {"corpus_id": 0, "score": 1.0}]
        )

    with CrossEncoderServiceClient(
        "https://models.test", transport=httpx.MockTransport(respond)
    ) as client:
        assert client.rank("q", ["first", "second"], top_k=1) == [
            {"corpus_id": 1, "score": 2.0}
        ]
    assert requests[0].url.path == RERANK_ROUTE
    assert json.loads(requests[0].content) == {
        "query": "q",
        "texts": ["first", "second"],
    }


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=[{"corpus_id": 5, "score": 1}]),
    ],
)
def test_remote_reranker_failures_use_adapter_error(response):
    with CrossEncoderServiceClient(
        "https://models.test", transport=httpx.MockTransport(lambda _: response)
    ) as client:
        with pytest.raises(ModelServiceError):
            client.rank("q", ["only document"])
