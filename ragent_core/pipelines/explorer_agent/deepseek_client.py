import asyncio
import json
import logging
import os
from typing import Any, Optional

import backoff
from openai import AsyncOpenAI

from ragent_core.types import QA

logger = logging.getLogger(__name__)


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_tool",
            "description": "Search the corpus for relevant documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_tool",
            "description": "Read the content of one or more documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of document IDs. If more than 3 IDs are provided, only the first 3 will be processed.",
                    },
                },
                "required": ["doc_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_scan_tool",
            "description": "Scan all documents for regex or fixed-string substring matches",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "fixed_string": {"type": "boolean", "default": True},
                    "case_sensitive": {"type": "boolean", "default": False},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_qa_pair_tool",
            "description": "Submit a QA pair to the pipeline",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "doc_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["question", "answer", "doc_ids"],
            },
        },
    },
]


def submit_qa_pair_tool(question: str, answer: str, doc_ids: list[int]):
    """Submit a QA pair to the pipeline."""
    logger.info(
        "Submit QA pair tool called with question: %s, answer preview: %s... with doc_ids: %s",
        question,
        answer[:100],
        doc_ids,
    )
    return {"status": "success"}


class DeepseekClient:
    def __init__(self, retriever, provider: str = "deepseek") -> None:
        """
        Initialize DeepseekClient for Deepseek or Xiaomi provider.

        Args:
            retriever: The retriever instance for tool calls
            provider: Either "deepseek" or "xiaomi" to determine API configuration
        """
        provider_lower = provider.strip().lower()
        if provider_lower == "xiaomi":
            self._provider = "xiaomi"
            base_url = os.environ.get("MIMO_BASE_URL")
            api_key = os.environ.get("MIMO_API_KEY")
        else:
            self._provider = "deepseek"
            base_url = os.environ.get("DEEPSEEK_BASE_URL")
            api_key = os.environ.get("DEEPSEEK_API_KEY")

        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=300,  # 5 minutes,
            max_retries=3,
        )
        self._retriever = retriever
        self._tool_call_map = {
            "search_tool": retriever.search_tool,
            "read_tool": retriever.read_tool,
            "text_scan_tool": retriever.text_scan_tool,
            "submit_qa_pair_tool": submit_qa_pair_tool,
        }

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter
    )
    async def _chat_completion_with_retry(
        self,
        model: str,
        messages: list[dict[str, Any]],
        semaphore: asyncio.Semaphore,
        system_prompt: Optional[str] = None,
        use_tools: bool = False,
    ):
        """Invoke the chat completion API with optional post-processing and retries."""

        async with semaphore:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }

            # Apply provider-specific parameters
            if self._provider == "xiaomi":
                kwargs["temperature"] = 0.3
                kwargs["top_p"] = 0.95
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

            if system_prompt and messages[0]["role"] != "system":
                kwargs["messages"].insert(
                    0, {"role": "system", "content": system_prompt}
                )
            if use_tools:
                kwargs["tools"] = tools

            response = await self._client.chat.completions.create(**kwargs)

            return response

    async def chat_completion_with_retry(
        self,
        model: str,
        content: str,
        semaphore: asyncio.Semaphore,
        system_prompt: Optional[str] = None,
        use_tools: bool = False,
    ) -> str:
        """Invoke the chat completion API with optional post-processing and retries."""
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        response = await self._chat_completion_with_retry(
            model,
            messages,
            semaphore,
            system_prompt,
            use_tools,
        )
        return response.choices[0].message.content

    def extract_qa_pairs(self, messages) -> list[QA]:
        qa_pairs = []
        for message in messages:
            if (
                not isinstance(message, dict)
                and message.role == "assistant"
                and message.tool_calls
            ):
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "submit_qa_pair_tool":
                        qa_data = json.loads(tool_call.function.arguments)
                        qa_pair = QA(
                            question=qa_data["question"],
                            answer=qa_data["answer"],
                            doc_indices=qa_data["doc_ids"] or [],
                        )
                        qa_pairs.append(qa_pair)
        return qa_pairs

    async def agent_loop(
        self,
        model: str,
        content: str,
        semaphore: asyncio.Semaphore,
        system_prompt: Optional[str] = None,
        max_tool_calls: int = 100,
    ) -> list[dict[str, Any]]:
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        tool_call_count = 0

        while True:
            response = await self._chat_completion_with_retry(
                model=model,
                messages=messages,
                semaphore=semaphore,
                system_prompt=system_prompt,
                use_tools=True,
            )
            if response is None:
                logger.warning("Chat completion returned None; breaking turn loop")
                break

            messages.append(response.choices[0].message)
            tool_calls = response.choices[0].message.tool_calls

            # If there are no tool calls, then the model has a final answer
            if tool_calls is None:
                logger.info(f"Model response: {response.choices[0].message.content}")
                break

            for tool in tool_calls:
                # Check if we've reached the max tool calls limit
                if tool_call_count >= max_tool_calls:
                    logger.warning(
                        "Reached maximum tool calls limit (%d); stopping agent loop",
                        max_tool_calls,
                    )
                    break

                tool_function = self._tool_call_map[tool.function.name]
                tool_result = tool_function(**json.loads(tool.function.arguments))
                tool_call_count += 1
                logger.debug(
                    "%s tool called with arguments %s (call %d/%d)",
                    tool.function.name,
                    tool.function.arguments,
                    tool_call_count,
                    max_tool_calls,
                )
                logger.debug("Tool result: %s", tool_result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool.id,
                        "content": json.dumps(tool_result)
                        if isinstance(tool_result, dict)
                        else tool_result,
                    }
                )

            # Check again after processing all tool calls in this response
            if tool_call_count >= max_tool_calls:
                break

        return messages

    async def run_agent(
        self,
        content: str,
        model: str,
        semaphore: asyncio.Semaphore,
        system_prompt: Optional[str] = None,
        max_tool_calls: int = 100,
    ) -> list[QA]:
        """Run an agentic turn with tool calls until the model stops calling tools.

        Args:
            content: The user message content.
            model: The model name to use.
            semaphore: Semaphore for concurrency control.
            system_prompt: Optional system prompt.
            max_tool_calls: Maximum number of tool calls allowed per invocation (default: 100).
        """
        messages = await self.agent_loop(
            model, content, semaphore, system_prompt, max_tool_calls
        )
        return self.extract_qa_pairs(messages)
