import math
import os
from dataclasses import dataclass

RETRYABLE_STATUS_CODES = frozenset({408, 429, 502, 503, 504})


@dataclass(frozen=True)
class ServiceClientConfig:
    timeout: float = 60.0
    connect_timeout: float = 5.0
    max_retries: int = 2
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "timeout",
            "connect_timeout",
            "retry_base_seconds",
            "retry_max_seconds",
        ):
            value = getattr(self, name)
            if (
                not math.isfinite(value)
                or value < 0
                or (name in {"timeout", "connect_timeout"} and value == 0)
            ):
                raise ValueError(
                    f"RAGENT_MODEL_SERVICE_{name.upper()} has an invalid value"
                )
        if self.max_retries < 0:
            raise ValueError("RAGENT_MODEL_SERVICE_MAX_RETRIES cannot be negative")
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("Retry maximum must be at least the base delay")

    @classmethod
    def from_environment(cls) -> "ServiceClientConfig":
        defaults = cls()
        return cls(
            **{
                name: (int if name == "max_retries" else float)(
                    os.getenv(
                        f"RAGENT_MODEL_SERVICE_{name.upper()}",
                        str(getattr(defaults, name)),
                    )
                )
                for name in cls.__dataclass_fields__
            }
        )
