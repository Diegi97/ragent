"""The Verifiers token-in/token-out generate API over Fireworks snapshots.

Verifiers retains behavior logprobs in its graph. Raw logprobs and Fireworks R3
references stay here until the completed episode is converted for training.
"""

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import verifiers.v1 as vf
from aiohttp import web
from fireworks.training.sdk.sampling import SampledCompletion
from training.utils.rl.async_rl.errors import classify_rollout_error
from training.utils.rl.rollout.renderer import sampled_completion_to_rollout_run
from training.utils.rl.rollout.types import (
    validate_rollout_run_routing,
)

logger = logging.getLogger(__name__)

ROLLOUT_ID = "ragent_rollout_id"
_SAMPLING_KEYS = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
    }
)
_WIRE_KEYS = frozenset(
    {"logprobs", "skip_special_tokens", "routed_experts_prompt_start"}
)


class TrainingEvidenceError(ValueError):
    """A rollout cannot be trained without inventing token-level evidence."""


def completion_logprobs(completion: SampledCompletion, name: str) -> list[float]:
    values = getattr(completion, name)
    count = len(completion.full_tokens) - completion.prompt_len
    if values is None:
        if not count:
            return []
        raise TrainingEvidenceError(f"Fireworks omitted {name}")
    values = list(values)
    if completion.logprobs_echoed:
        if len(values) != len(completion.full_tokens) - 1:
            raise TrainingEvidenceError(f"Echoed {name} does not align with tokens")
        values = values[completion.prompt_len - 1 :]
    if len(values) != count or any(
        isinstance(v, bool)
        or v is None
        or not math.isfinite(float(v))
        or float(v) == -9999.0
        for v in values
    ):
        raise TrainingEvidenceError(f"Missing, nonfinite, or misaligned {name}")
    return [float(v) for v in values]


@dataclass(frozen=True)
class CapturedTurn:
    trace_id: str
    request_id: str
    completion: SampledCompletion

    @property
    def prompt_ids(self) -> list[int]:
        return self.completion.full_tokens[: self.completion.prompt_len]

    @property
    def completion_ids(self) -> list[int]:
        return self.completion.full_tokens[self.completion.prompt_len :]


@dataclass
class RolloutCapture:
    lease: Any
    turns: list[CapturedTurn] = field(default_factory=list)
    fatal_error: Exception | None = None
    handlers: set[asyncio.Task] = field(default_factory=set)


class FireworksGenerateServer:
    def __init__(
        self, sample_kwargs: dict[str, Any], *, host: str = "127.0.0.1", port: int = 0
    ) -> None:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("The generate bridge must bind to loopback")
        if not 0 <= port <= 65535:
            raise ValueError("bridge_port must be in [0, 65535]")
        self.sample_kwargs = dict(sample_kwargs)
        self.host = host
        self.port = port
        self.base_url = ""
        self._runner: web.AppRunner | None = None
        self.captures: dict[str, RolloutCapture] = {}

    async def start(self) -> None:
        if self._runner is not None:
            return
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_post("/inference/v1/generate", self._handle_generate)
        app.router.add_get("/v1/models", self._handle_models)
        runner = web.AppRunner(app, handler_cancellation=True)
        await runner.setup()
        try:
            await web.TCPSite(runner, self.host, self.port).start()
        except BaseException:
            await runner.cleanup()
            raise
        self._runner = runner
        host, port = runner.addresses[0][:2]
        host = f"[{host}]" if ":" in host else host
        self.base_url = f"http://{host}:{port}/v1"
        logger.info("Fireworks generate bridge listening on %s", self.base_url)

    async def stop(self) -> None:
        for rollout_id in list(self.captures):
            await self.release(rollout_id)
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self.base_url = ""

    def train_client_config(
        self, renderer_model_name: str, renderer: Any = None
    ) -> vf.TrainClientConfig:
        if not self.base_url:
            raise RuntimeError("The generate bridge is not running")
        return vf.TrainClientConfig(
            base_url=self.base_url,
            api_key_var="RAGENT_BRIDGE_API_KEY",
            renderer_model_name=renderer_model_name,
            renderer=renderer,
        )

    def register(self, lease: Any) -> str:
        rollout_id = uuid.uuid4().hex
        self.captures[rollout_id] = RolloutCapture(lease)
        return rollout_id

    async def release(self, rollout_id: str) -> None:
        capture = self.captures.pop(rollout_id, None)
        if capture is not None:
            pending = list(capture.handlers)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _handle_models(self, request: web.Request) -> web.Response:
        del request
        models = {c.lease.model for c in self.captures.values()}
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": model,
                        "object": "model",
                        "max_model_len": self.sample_kwargs["max_seq_len"],
                    }
                    for model in sorted(models)
                ],
            }
        )

    async def _handle_generate(self, request: web.Request) -> web.Response:
        capture: RolloutCapture | None = None
        task = asyncio.current_task()
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("Generate request must be an object")
            params = body.get("sampling_params") or {}
            if not isinstance(params, dict):
                raise ValueError("sampling_params must be an object")
            rollout_id = params.get(ROLLOUT_ID)
            if not isinstance(rollout_id, str) or rollout_id not in self.captures:
                raise ValueError("Unknown or missing training rollout identifier")
            capture = self.captures[rollout_id]
            if task is not None:
                capture.handlers.add(task)
            prompt = body.get("token_ids")
            if (
                not isinstance(prompt, list)
                or not prompt
                or any(type(token) is not int or token < 0 for token in prompt)
            ):
                raise ValueError("token_ids must be a nonempty list of integer IDs")
            if body.get("features"):
                raise ValueError("Multimodal generate requests are not supported")
            if body.get("model") != capture.lease.model:
                raise ValueError("Request model differs from the pinned snapshot")
            trace_id = request.headers.get("X-Session-ID")
            if not trace_id:
                raise ValueError("Generate request must include X-Session-ID")
            unsupported = (
                set(params)
                - _SAMPLING_KEYS
                - _WIRE_KEYS
                - {ROLLOUT_ID, "stop_token_ids"}
            )
            if unsupported:
                raise ValueError(
                    f"Unsupported sampling settings: {sorted(unsupported)}"
                )
            kwargs = dict(self.sample_kwargs)
            kwargs.update(
                {key: val for key, val in params.items() if key in _SAMPLING_KEYS}
            )
            if kwargs.get("top_k") == -1:
                kwargs["top_k"] = (
                    0  # vLLM and Fireworks spell disabled top-k differently.
                )
            stops = params.get("stop_token_ids") or []
            if not isinstance(stops, list) or any(type(t) is not int for t in stops):
                raise ValueError("stop_token_ids must be a list of integer IDs")
            kwargs.update(n=1, logprobs=True, stop=stops or None)
            cap = kwargs["max_seq_len"]
            if len(prompt) >= cap:
                return web.json_response(
                    {"error": {"message": "Prompt exceeds the model context length"}},
                    status=400,
                )
            kwargs["max_tokens"] = min(kwargs["max_tokens"], cap - len(prompt))
            completions = await capture.lease.sample_with_prompt_tokens(
                prompt, **kwargs
            )
            if len(completions) != 1:
                raise TrainingEvidenceError(
                    "Fireworks must return exactly one completion"
                )
            completion = completions[0]
            if completion.full_tokens[: completion.prompt_len] != prompt:
                raise TrainingEvidenceError("Fireworks changed the prompt token IDs")
            if (
                completion.prompt_len != len(prompt)
                or any(type(t) is not int or t < 0 for t in completion.full_tokens)
                or len(completion.full_tokens) > cap
            ):
                raise TrainingEvidenceError("Invalid Fireworks token sequence")
            behavior = completion_logprobs(completion, "sampling_logprobs")
            completion_logprobs(completion, "inference_logprobs")
            if behavior and kwargs.get("include_routing_matrix"):
                run = sampled_completion_to_rollout_run(completion, reward=0.0)
                if run is None or completion.routing_matrices is None:
                    raise TrainingEvidenceError(
                        "Fireworks omitted requested router replay"
                    )
                validate_rollout_run_routing(run)
            request_id = uuid.uuid4().hex
            turn = CapturedTurn(trace_id, request_id, completion)
            capture.turns.append(turn)
            return web.json_response(
                {
                    "request_id": request_id,
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": turn.completion_ids,
                            "finish_reason": completion.finish_reason,
                            "logprobs": {
                                "content": [
                                    {"token": f"token_id:{token}", "logprob": lp}
                                    for token, lp in zip(
                                        turn.completion_ids, behavior, strict=True
                                    )
                                ]
                            },
                        }
                    ],
                }
            )
        except Exception as error:
            classified = classify_rollout_error(error)
            if capture is not None and not classified.recoverable:
                capture.fatal_error = error
            status = classified.status_code or (502 if classified.recoverable else 400)
            logger.warning("Generate request failed (%s): %s", status, error)
            return web.json_response(
                {"error": {"message": str(error), "type": type(error).__name__}},
                status=status,
            )
        finally:
            if capture is not None and task is not None:
                capture.handlers.discard(task)
