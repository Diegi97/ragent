from enum import StrEnum


class ProviderFailureKind(StrEnum):
    TRANSPORT = "transport"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    API = "api"
    INVALID_RESPONSE = "invalid_response"

    @classmethod
    def for_status(cls, status: int) -> "ProviderFailureKind":
        if status in {401, 403}:
            return cls.AUTHENTICATION
        if status == 429:
            return cls.RATE_LIMIT
        return cls.API


class ProviderError(RuntimeError):
    """Stable provider failure whose message excludes vendor bodies and URLs."""

    def __init__(
        self, provider: str, kind: ProviderFailureKind, status_code: int | None = None
    ):
        self.provider = provider
        self.kind = kind
        self.status_code = status_code
        detail = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"{provider} {kind.value} failure{detail}")
