import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import turbopuffer
from turbopuffer.types.namespace_multi_query_response import (
    NamespaceMultiQueryResponse,
)
from turbopuffer.types.namespace_query_response import NamespaceQueryResponse

from ragent_core.provider_errors import ProviderError, ProviderFailureKind
from ragent_core.retrievers.settings import DEFAULT_TURBOPUFFER_REGION, REGION_ENV

REQUEST_TIMEOUT_SECONDS = 60.0
MAX_RETRIES = 4


def create_turbopuffer_client(
    *,
    api_key: str | None = None,
    region: str | None = None,
) -> Any:
    resolved_region = region or os.getenv(REGION_ENV) or DEFAULT_TURBOPUFFER_REGION
    options: dict[str, Any] = {
        "region": resolved_region,
        "timeout": REQUEST_TIMEOUT_SECONDS,
        "max_retries": MAX_RETRIES,
    }
    if api_key is not None:
        options["api_key"] = api_key
    return turbopuffer.Turbopuffer(**options)


def query_rows(response: NamespaceQueryResponse) -> list[dict[str, Any]]:
    return [row.model_dump(by_alias=True) for row in response.rows or []]


def multi_query_rows(
    response: NamespaceMultiQueryResponse,
) -> list[dict[str, Any]]:
    return [
        row.model_dump(by_alias=True)
        for result in response.results
        for row in result.rows or []
    ]


def create_async_turbopuffer_client(
    *, region: str | None = None
) -> turbopuffer.AsyncTurbopuffer:
    return turbopuffer.AsyncTurbopuffer(
        region=region or os.getenv(REGION_ENV) or DEFAULT_TURBOPUFFER_REGION,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )


class RetrievalServiceError(ProviderError):
    def __init__(self, kind: ProviderFailureKind, status_code: int | None = None):
        super().__init__("Turbopuffer", kind, status_code)


@contextmanager
def turbopuffer_errors() -> Iterator[None]:
    """Translate SDK failures before they enter pipeline, tool or IPC contracts."""
    try:
        yield
    except turbopuffer.APIConnectionError as exc:
        raise RetrievalServiceError(ProviderFailureKind.TRANSPORT) from exc
    except turbopuffer.APIStatusError as exc:
        raise RetrievalServiceError(
            ProviderFailureKind.for_status(exc.status_code), exc.status_code
        ) from exc
    except turbopuffer.APIError as exc:
        raise RetrievalServiceError(ProviderFailureKind.API) from exc
