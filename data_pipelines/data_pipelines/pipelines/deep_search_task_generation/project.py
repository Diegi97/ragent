import os
from datetime import datetime, timezone

PROJECT_NAME = "deep-search-task-generation"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def phoenix_project() -> str:
    return os.getenv("PHOENIX_DEEP_SEARCH_TASK_GENERATION_PROJECT_NAME", PROJECT_NAME)
