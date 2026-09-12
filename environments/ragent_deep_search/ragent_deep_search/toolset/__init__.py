import asyncio

import verifiers.v1 as vf
from pydantic import SkipValidation

from ragent_core.retrievers.agent_retriever import AgentRetriever
from ragent_core.retrievers.tool_protocol import (
    DEFAULT_SCAN_RESULTS,
    DEFAULT_SCAN_SNIPPET_CHARS,
    ToolName,
)
from ragent_deep_search.toolset.config import RagentState, RagentToolsetConfig


class RagentToolset(vf.Toolset[RagentToolsetConfig, RagentState]):
    # This taskset owns a single tool server, so bare names cannot collide.
    TOOL_PREFIX = None

    @classmethod
    def for_launch(cls, config: RagentToolsetConfig) -> "RagentToolset":
        """Resolve local paths before Verifiers changes the tool server directory."""
        env_file = config.env_file
        if env_file is not None:
            env_file = env_file.expanduser().resolve()
        return cls(config.model_copy(update={"env_file": env_file}))

    async def setup(self) -> None:
        """Load the namespace-level retriever once per environment worker."""
        self.retriever = await asyncio.to_thread(self._build_retriever)

    async def search(self, queries: list[str], *, table_name: str) -> str:
        """Call the search implementation directly for a known corpus."""
        return await asyncio.to_thread(
            self.retriever.search_tool,
            queries,
            table_name,
        )

    async def read(
        self,
        doc_ids: list[SkipValidation[int]],
        *,
        table_name: str,
    ) -> str:
        """Call the document reader directly for a known corpus."""
        validation_error = self._read_validation_error(doc_ids)
        if validation_error is not None:
            return validation_error
        return await asyncio.to_thread(
            self.retriever.read_tool,
            doc_ids,
            table_name,
        )

    @vf.tool(name=ToolName.SEARCH)
    async def search_tool(self, queries: list[str]) -> str:
        """Search the active corpus for documents relevant to up to three queries."""
        return await self.search(queries, table_name=self._table_name())

    @vf.tool(name=ToolName.READ)
    # Skip validation so numeric strings reach the tool and get a friendly error.
    async def read_tool(self, doc_ids: list[SkipValidation[int]]) -> str:
        """Read up to three documents by integer ID; numeric strings are rejected."""
        return await self.read(doc_ids, table_name=self._table_name())

    @vf.tool(name=ToolName.TEXT_SCAN)
    async def text_scan_tool(
        self,
        pattern: str,
        fixed_string: bool = True,
        case_sensitive: bool = False,
        max_results: int = DEFAULT_SCAN_RESULTS,
        snippet_chars: int = DEFAULT_SCAN_SNIPPET_CHARS,
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

    def _build_retriever(self) -> AgentRetriever:
        return AgentRetriever.from_turbopuffer_index(
            namespace=self.config.namespace,
            device=self.config.device,
            retrieval_mode=self.config.retrieval_mode,
            turbopuffer_api_key=self.config.turbopuffer_api_key(),
        )

    def _table_name(self) -> str:
        table_name = self.state.table_name
        if not table_name:
            raise RuntimeError("No corpus was assigned to this rollout.")
        return table_name

    @staticmethod
    def _read_validation_error(
        doc_ids: list[SkipValidation[int]],
    ) -> str | None:
        if any(
            not isinstance(doc_id, int) or isinstance(doc_id, bool)
            for doc_id in doc_ids
        ):
            return (
                "Error: The read tool accepts only integers as document IDs. "
                'Retry with integer IDs, for example: {"doc_ids": [2696, 2808]}.'
            )
        return None
