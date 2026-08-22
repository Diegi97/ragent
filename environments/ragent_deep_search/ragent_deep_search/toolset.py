# servers/ragent.py
import asyncio
import os
from pathlib import Path

import verifiers.v1 as vf
from dotenv import dotenv_values
from pydantic import SkipValidation

from ragent_core.retrievers import AgentRetriever, RetrievalMode


class RagentState(vf.State):
    # A default is required because verifiers constructs the state before Task.setup().
    table_name: str = ""


class RagentToolsetConfig(vf.SharedToolsetConfig):
    namespace: str = "default"
    device: str | None = None
    retrieval_mode: RetrievalMode = RetrievalMode.BM25
    env_file: Path | None = None


class RagentToolset(vf.Toolset[RagentToolsetConfig, RagentState]):
    # This taskset owns a single tool server, so bare names cannot collide.
    TOOL_PREFIX = None

    # Weird workaround for Verifiers changing the tool server's working directory.
    @classmethod
    def for_launch(cls, config: RagentToolsetConfig) -> "RagentToolset":
        """Resolve local paths before Verifiers changes the tool server directory."""
        env_file = config.env_file
        if env_file is not None:
            env_file = env_file.expanduser().resolve()
        return cls(config.model_copy(update={"env_file": env_file}))

    def _turbopuffer_api_key(self) -> str:
        api_key = os.getenv("TURBOPUFFER_API_KEY")
        if api_key is None and self.config.env_file is not None:
            value = dotenv_values(self.config.env_file).get("TURBOPUFFER_API_KEY")
            if isinstance(value, str):
                api_key = value
        if not api_key:
            raise ValueError(
                "TURBOPUFFER_API_KEY is required by the retrieval tools. "
                "Set env.taskset.tools.env_file to an uncommitted dotenv file "
                "containing it."
            )
        return api_key

    async def setup(self) -> None:
        """Load the namespace-level retriever once per environment worker."""
        self.retriever = AgentRetriever.from_turbopuffer_index(
            namespace=self.config.namespace,
            device=self.config.device,
            retrieval_mode=self.config.retrieval_mode,
            turbopuffer_api_key=self._turbopuffer_api_key(),
        )

    def _table_name(self) -> str:
        table_name = self.state.table_name
        if not table_name:
            raise RuntimeError("No corpus was assigned to this rollout.")
        return table_name

    @vf.tool(name="search")
    async def search_tool(self, queries: list[str]) -> str:
        """Search the active corpus for documents relevant to up to three queries."""
        table_name = self._table_name()
        return await asyncio.to_thread(
            self.retriever.search_tool,
            queries,
            table_name,
        )

    @vf.tool(name="read")
    # Skip validation so numeric strings reach the tool and get a friendly error.
    async def read_tool(self, doc_ids: list[SkipValidation[int]]) -> str:
        """Read up to three documents by integer ID; numeric strings are rejected."""
        if any(
            not isinstance(doc_id, int) or isinstance(doc_id, bool)
            for doc_id in doc_ids
        ):
            return (
                "Error: The read tool accepts only integers as document IDs. "
                'Retry with integer IDs, for example: {"doc_ids": [2696, 2808]}.'
            )

        table_name = self._table_name()
        return await asyncio.to_thread(
            self.retriever.read_tool,
            doc_ids,
            table_name,
        )

    @vf.tool(name="text_scan")
    async def text_scan_tool(
        self,
        pattern: str,
        fixed_string: bool = True,
        case_sensitive: bool = False,
        max_results: int = 25,
        snippet_chars: int = 200,
    ) -> str:
        """Scan the active corpus for a fixed string or regular expression."""
        table_name = self._table_name()
        return await asyncio.to_thread(
            self.retriever.text_scan_tool,
            pattern,
            table_name,
            fixed_string,
            case_sensitive,
            max_results,
            snippet_chars,
        )


if __name__ == "__main__":
    RagentToolset.run()
