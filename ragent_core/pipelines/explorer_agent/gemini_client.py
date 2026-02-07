import asyncio
import logging
from typing import List, Optional

import backoff
from dotenv import load_dotenv
from google import genai
from google.genai import types

from ragent_core.types import QA

logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self, retriever) -> None:
        load_dotenv()
        self._client = genai.Client(
            http_options=types.HttpOptions(
                timeout=45 * 1000,  # 45 seconds in milliseconds
                retry_options=types.HttpRetryOptions(
                    attempts=5,
                ),
            ),
        ).aio
        self._retriever = retriever

    def search_tool(self, queries: List[str]) -> str:
        """
        Search corpus using BM25 + ColBERT (RRF) with mxbai-rerank-v2 reranking.

        Args:
            queries: List of search query strings.

        Returns:
            XML-formatted string with search results grouped by query.
        """
        return self._retriever.search_tool(queries)

    def text_scan_tool(
        self,
        pattern: str,
        fixed_string: bool = True,
        case_sensitive: bool = False,
        max_results: int = 25,
        snippet_chars: int = 200,
    ) -> str:
        """
        Scan all documents for regex or fixed-string matches, ranked by match count.

        Args:
            pattern: Search pattern (literal string or regex).
            fixed_string: If True, treat as literal substring; if False, as regex.
            case_sensitive: Whether matching is case-sensitive.
            max_results: Maximum matches to return.
            snippet_chars: Characters around match to include in snippet.

        Returns:
            XML string: <match><id>...</id><title>...</title><snippet>...</snippet></match>...
        """
        return self._retriever.text_scan_tool(
            pattern, fixed_string, case_sensitive, max_results, snippet_chars
        )

    def read_tool(self, doc_ids: List[int]) -> str:
        """
        Retrieve document text by IDs.

        Args:
            doc_ids: List of document IDs (integers).

        Returns:
            String: The content of the documents as an XML string.
        """
        return self._retriever.read_tool(doc_ids)

    def submit_qa_pair_tool(self, question: str, answer: str, doc_ids: List[int]):
        """
        Submit a verified QA pair to the pipeline.

        Args:
            question: Atomic question string.
            answer: Answer derived from read documents.
            doc_ids: List of document IDs that support the answer.

        Returns:
            Dict with "status": "success".
        """
        logger.info(
            "Submit QA pair tool called with question: %s, answer preview: %s... with doc_ids: %s",
            question,
            answer[:100],
            doc_ids,
        )
        return {"status": "success"}

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter
    )
    async def _chat_completion_with_retry(
        self,
        content: str,
        model: str,
        semaphore: asyncio.Semaphore,
        system_prompt: Optional[str] = None,
        use_tools: bool = False,
    ):
        """Invoke Gemini via ADK with optional post-processing and retries."""
        async with semaphore:
            config = types.GenerateContentConfig(
                tools=[
                    self.search_tool,
                    self.text_scan_tool,
                    self.read_tool,
                    self.submit_qa_pair_tool,
                ]
                if use_tools
                else None,
                system_instruction=system_prompt or None,
            )
            config.automatic_function_calling = types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=100
            )
            if model == "gemini-2.5-flash-lite":
                config.thinking_config = types.ThinkingConfig(thinking_budget=-1) # automatic thinking budget

            response = await self._client.models.generate_content(
                model=model, contents=content, config=config
            )
            return response

    async def chat_completion_with_retry(
        self,
        content: str,
        model: str,
        semaphore: asyncio.Semaphore,
        system_prompt: Optional[str] = None,
        use_tools: bool = False,
    ) -> str:
        response = await self._chat_completion_with_retry(
            content, model, semaphore, system_prompt, use_tools
        )
        return response.text or ""

    def extract_qa_pairs(self, response) -> list[QA]:
        pairs = []
        for content in response.automatic_function_calling_history:
            for part in content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "submit_qa_pair_tool"
                ):
                    pairs.append(
                        QA(
                            question=part.function_call.args.get("question"),
                            answer=part.function_call.args.get("answer"),
                            doc_ids=part.function_call.args.get("doc_ids", []),
                            info={},
                        )
                    )
        return pairs

    async def run_agent(
        self,
        content: str,
        model: str,
        semaphore: asyncio.Semaphore,
        system_prompt: Optional[str] = None,
    ) -> list[QA]:
        """Run an agentic turn with tool calls until the model completes."""
        response = await self._chat_completion_with_retry(
            content, model, semaphore, system_prompt, use_tools=True
        )
        return self.extract_qa_pairs(response)
