from typing import Any

import httpx
from httpx_retries import Retry, retry_request

from ragent_core.retrievers.model_contract import ModelServiceError
from ragent_core.retrievers.service_clients.config import (
    RETRYABLE_STATUS_CODES,
    ServiceClientConfig,
)


class ModelServiceClient:
    def __init__(
        self,
        base_url: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("Model service base_url cannot be empty")
        config = ServiceClientConfig.from_environment()
        self.base_url = base_url.rstrip("/")
        self._retry = Retry(
            total=config.max_retries,
            max_backoff_wait=config.retry_max_seconds,
            backoff_factor=config.retry_base_seconds,
            respect_retry_after_header=True,
            allowed_methods={"POST"},
            status_forcelist=RETRYABLE_STATUS_CODES,
            backoff_jitter=1.0,
        )
        self._client = httpx.Client(
            timeout=httpx.Timeout(config.timeout, connect=config.connect_timeout),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ModelServiceClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            response = retry_request(
                self._client,
                "POST",
                f"{self.base_url}{path}",
                retry=self._retry,
                json=payload,
            )
            try:
                response.raise_for_status()
                try:
                    return response.json()
                except ValueError as exc:
                    raise ModelServiceError(
                        f"Model service returned non-JSON data for {path}"
                    ) from exc
            finally:
                response.close()
        except httpx.HTTPStatusError as exc:
            raise ModelServiceError(
                f"Model service returned HTTP {exc.response.status_code} for {path}"
            ) from exc
        except httpx.RequestError as exc:
            raise ModelServiceError(
                f"Model service transport failed for {path}"
            ) from exc
