from types import SimpleNamespace

import httpx
import openai
import pytest

from data_pipelines.providers import fireworks
from data_pipelines.providers import openai as completions
from ragent_core.provider_errors import ProviderError, ProviderFailureKind


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,kind",
    [
        (None, ProviderFailureKind.TRANSPORT),
        (401, ProviderFailureKind.AUTHENTICATION),
        (429, ProviderFailureKind.RATE_LIMIT),
        (500, ProviderFailureKind.API),
    ],
)
async def test_completion_sdk_errors_are_normalized(monkeypatch, status, kind):
    request = httpx.Request("POST", "https://provider.test")
    sdk_error = (
        openai.APIConnectionError(message="vendor-private-detail", request=request)
        if status is None
        else openai.APIStatusError(
            "vendor-private-detail",
            response=httpx.Response(status, request=request),
            body={},
        )
    )

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.complete)
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def complete(self, **kwargs):
            raise sdk_error

    monkeypatch.setattr(completions, "AsyncOpenAI", Client)
    with pytest.raises(ProviderError) as failure:
        await completions.chat_completion([{"role": "user", "content": "q"}], "fake")
    assert failure.value.kind is kind
    assert failure.value.__cause__ is sdk_error
    assert "vendor-private-detail" not in str(failure.value)


@pytest.mark.parametrize(
    "kind",
    [
        ProviderFailureKind.TRANSPORT,
        ProviderFailureKind.AUTHENTICATION,
        ProviderFailureKind.RATE_LIMIT,
        ProviderFailureKind.INVALID_RESPONSE,
    ],
)
def test_fireworks_failures_are_normalized(fireworks_transport, tmp_path, kind):
    def handler(request):
        if kind is ProviderFailureKind.TRANSPORT:
            raise httpx.ConnectError("vendor-private-detail", request=request)
        if kind is ProviderFailureKind.INVALID_RESPONSE:
            return httpx.Response(200, text="vendor-private-detail")
        return httpx.Response(
            401 if kind is ProviderFailureKind.AUTHENTICATION else 429,
            text="vendor-private-detail",
        )

    fireworks_transport(handler)
    with pytest.raises(fireworks.FireworksError) as failure:
        fireworks.download_batch_dataset("dataset", tmp_path, 1)
    assert failure.value.kind is kind
    assert "vendor-private-detail" not in str(failure.value)
    assert list(tmp_path.iterdir()) == []
