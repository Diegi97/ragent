import os
from pathlib import Path

import verifiers.v1 as vf
from dotenv import dotenv_values

from ragent_core.retrievers.mode import RetrievalMode
from ragent_core.retrievers.settings import (
    DEFAULT_LOGICAL_NAMESPACE,
    TURBOPUFFER_API_KEY_ENV,
)


class RagentState(vf.State):
    # A default is required because verifiers constructs the state before Task.setup().
    table_name: str = ""


class RagentToolsetConfig(vf.SharedToolsetConfig):
    namespace: str = DEFAULT_LOGICAL_NAMESPACE
    device: str | None = None
    retrieval_mode: RetrievalMode = RetrievalMode.BM25
    env_file: Path | None = None

    def turbopuffer_api_key(self) -> str:
        api_key = os.getenv(TURBOPUFFER_API_KEY_ENV)
        if api_key is None and self.env_file is not None:
            value = dotenv_values(self.env_file).get(TURBOPUFFER_API_KEY_ENV)
            if isinstance(value, str):
                api_key = value
        if not api_key:
            raise ValueError(
                "TURBOPUFFER_API_KEY is required by the retrieval tools. "
                "Set env.taskset.tools.env_file to an uncommitted dotenv file "
                "containing it."
            )
        return api_key
