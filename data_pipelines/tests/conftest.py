import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider

from data_pipelines.providers import fireworks
from data_pipelines.providers.config import OPENAI_API_KEY_ENV
from data_pipelines.tracing import configure_tracing


@pytest.fixture
def local_tracing(monkeypatch):
    project = "test-generation"
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", project)
    monkeypatch.setenv("PHOENIX_DEEP_SEARCH_TASK_GENERATION_PROJECT_NAME", project)
    provider = TracerProvider()
    configure_tracing(provider=provider, project_name=project)
    yield
    provider.shutdown()


@pytest.fixture
def fireworks_transport(monkeypatch):
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "account")
    monkeypatch.setenv(OPENAI_API_KEY_ENV, "fake-test-key")
    client_factory = httpx.Client

    def install(handler):
        client = client_factory(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(fireworks.httpx, "Client", lambda **kwargs: client)

    return install
