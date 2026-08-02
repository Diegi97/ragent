# servers/ragent.py
import asyncio

import verifiers.v1 as vf

from ragent_core.retrievers import AgentRetriever, RetrievalMode


class RagentState(vf.State):
    # A default is required because verifiers constructs the state before Task.setup().
    table_name: str = ""


class RagentToolsetConfig(vf.SharedToolsetConfig):
    namespace: str = "default"
    device: str | None = None
    retrieval_mode: RetrievalMode = RetrievalMode.BM25


class RagentToolset(vf.Toolset[RagentToolsetConfig, RagentState]):
    TOOL_PREFIX = "ragent"

    async def setup(self) -> None:
        """Load the namespace-level retriever once per environment worker."""
        self.retriever = AgentRetriever.from_lancedb_index(
            namespace=self.config.namespace,
            device=self.config.device,
            retrieval_mode=self.config.retrieval_mode,
        )

    def _table_name(self) -> str:
        table_name = self.state.table_name
        if not table_name:
            raise RuntimeError("No corpus was assigned to this rollout.")
        return table_name

    @vf.tool(name="search")
    async def search_tool(self, queries: list[str]) -> str:
        """Search the active corpus for documents relevant to one or more queries."""
        table_name = self._table_name()
        return await asyncio.to_thread(
            self.retriever.search_tool,
            queries,
            table_name,
        )

    @vf.tool(name="read")
    async def read_tool(self, doc_ids: list[int]) -> str:
        """Read up to three full documents from the active corpus by document ID."""
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
