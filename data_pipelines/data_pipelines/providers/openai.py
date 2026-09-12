import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError

from data_pipelines.providers.config import OPENAI_API_KEY_ENV
from ragent_core.provider_errors import ProviderError, ProviderFailureKind

DEFAULT_OPENAI_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_OPENAI_MAX_RETRIES = 5


class ModelResponseError(ValueError):
    """The completion provider did not return usable text."""


@dataclass(frozen=True)
class LLMCompletion:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_trace_attributes(self) -> dict[str, Any]:
        return {
            "llm.model_name": self.model,
            "llm.token_count.prompt": self.prompt_tokens,
            "llm.token_count.completion": self.completion_tokens,
            "llm.token_count.total": self.total_tokens,
        }

    @classmethod
    def from_response(cls, response: Any, requested_model: str) -> "LLMCompletion":
        if not response.choices:
            raise ModelResponseError("Completion response contains no choices")
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise ModelResponseError("Completion provider refused the request")
        if not isinstance(message.content, str) or not message.content.strip():
            raise ModelResponseError("Completion response contains no text")
        usage = response.usage
        return cls(
            content=message.content,
            model=response.model or requested_model,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=usage.completion_tokens if usage is not None else None,
            total_tokens=usage.total_tokens if usage is not None else None,
        )


async def chat_completion(
    messages: Sequence[dict[str, str]], model: str
) -> LLMCompletion:
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", str(DEFAULT_OPENAI_MAX_RETRIES)))
    if max_retries < 0:
        raise ValueError("OPENAI_MAX_RETRIES must be non-negative")
    try:
        async with AsyncOpenAI(
            api_key=os.getenv(OPENAI_API_KEY_ENV),
            base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            max_retries=max_retries,
        ) as client:
            response = await client.chat.completions.create(
                model=model, messages=list(messages)
            )
    except APIConnectionError as exc:
        raise ProviderError("Completion", ProviderFailureKind.TRANSPORT) from exc
    except APIStatusError as exc:
        raise ProviderError(
            "Completion",
            ProviderFailureKind.for_status(exc.status_code),
            exc.status_code,
        ) from exc
    except OpenAIError as exc:
        raise ProviderError("Completion", ProviderFailureKind.API) from exc
    return LLMCompletion.from_response(response, model)
