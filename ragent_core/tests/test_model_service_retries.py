import httpx
import httpx_retries.retry as retry_module
import pytest

from ragent_core.retrievers.model_contract import (
    EMBEDDING_QUERY_ROUTE,
    ModelServiceError,
)
from ragent_core.retrievers.service_clients.embedding import EmbeddingServiceClient


@pytest.mark.parametrize(
    "statuses,expected_attempts", [([503, 200], 2), ([503, 503, 503], 3), ([401], 1)]
)
def test_model_service_retry_policy_and_final_errors(
    monkeypatch, statuses, expected_attempts
):
    monkeypatch.setenv("RAGENT_MODEL_SERVICE_MAX_RETRIES", "2")
    monkeypatch.setenv("RAGENT_MODEL_SERVICE_RETRY_BASE_SECONDS", "0")
    monkeypatch.setenv("RAGENT_MODEL_SERVICE_RETRY_MAX_SECONDS", "5")
    monkeypatch.setenv("RAGENT_MODEL_SERVICE_TIMEOUT", "7")
    monkeypatch.setenv("RAGENT_MODEL_SERVICE_CONNECT_TIMEOUT", "3")
    sleeps = []
    monkeypatch.setattr(retry_module.time, "sleep", sleeps.append)
    requests = []

    def respond(request):
        requests.append(request)
        status = statuses[len(requests) - 1]
        return httpx.Response(status, headers={"Retry-After": "2"}, json=[[1.0, 2.0]])

    with EmbeddingServiceClient(
        "http://model.test", transport=httpx.MockTransport(respond)
    ) as client:
        if statuses[-1] == 200:
            assert client.encode_queries(["query"]).tolist() == [[1.0, 2.0]]
        else:
            with pytest.raises(ModelServiceError, match=f"HTTP {statuses[-1]}"):
                client.encode_queries(["query"])
    assert len(requests) == expected_attempts
    assert all(request.url.path == EMBEDDING_QUERY_ROUTE for request in requests)
    assert requests[0].extensions["timeout"]["read"] == 7
    assert requests[0].extensions["timeout"]["connect"] == 3
    assert sleeps == [2.0] * (expected_attempts - 1)
