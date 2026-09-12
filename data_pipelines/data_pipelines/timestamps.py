from datetime import datetime, timezone
from enum import StrEnum


class TimestampPrecision(StrEnum):
    SECONDS = "%Y%m%dT%H%M%SZ"
    MICROSECONDS = "%Y%m%dT%H%M%S%fZ"


def utc_timestamp(
    precision: TimestampPrecision = TimestampPrecision.SECONDS,
    created_at: datetime | None = None,
) -> str:
    instant = created_at or datetime.now(timezone.utc)
    return instant.astimezone(timezone.utc).strftime(precision.value)
