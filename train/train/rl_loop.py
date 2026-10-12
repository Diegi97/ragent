"""Train native Verifiers tasks on Fireworks' async serverless RL pool. Adapted from https://github.com/fw-ai/cookbook/blob/main/training/recipes/experiment/async_rl_loop_serverless.py"""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import signal
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import tinker
import verifiers.v1 as vf
from fireworks.training.sdk import FiretitanServiceClient
from fireworks.training.sdk.training_spec import (
    LRSchedulerSpec,
    compute_lr,
    default_constant_schedule,
    normalize_lr_scheduler_spec,
)
from renderers import config_from_name, create_renderer
from renderers.base import load_tokenizer
from training.train_loop import DynamicFilterFn
from training.utils import (
    DEFAULT_ADAM,
    WandBConfig,
    log_metrics,
    read_api_extra_headers_env,
    resolve_router_replay_enabled,
    setup_wandb,
    wandb_finish,
)
from training.utils.dataloader import CursorDataLoader
from training.utils.logging import ASYNC_RL_WANDB_METRIC_STEPS
from training.utils.rl.async_rl import (
    AsyncRLCoordinator,
    AsyncRLTelemetry,
    RolloutRow,
    TrainingChunk,
)
from training.utils.rl.grpo import make_grpo_loss_fn, validate_grpo_config
from training.utils.rl.losses import combine_prompt_groups
from training.utils.rl.metrics import datum_target_len
from training.utils.rl.rollout.types import RolloutRun
from training.utils.rl.router_replay import warn_if_full_sequence_router_replay
from training.utils.rl.tis import TISConfig
from training.utils.termination import TerminatedBySignal
from training.utils.timer import elapsed_timer, flush_timing, wall_timer

from train.environment import validate_environment
from train.fireworks_bridge import (
    ROLLOUT_ID,
    FireworksGenerateServer,
    TrainingEvidenceError,
)
from train.verifiers_rollout import episode_to_rollout_run

logger = logging.getLogger(__name__)

__all__ = [
    "Config",
    "ServerlessSampler",
    "main",
    "run_evaluation",
]

_MAX_SNAPSHOT_NAME_LENGTH = 54


def _generate_run_name() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"ragent-{timestamp}-{uuid.uuid4().hex[:12]}"


@dataclass
class Config:
    """Serverless async-RL configuration with no dedicated resource fields."""

    base_model: str = "accounts/fireworks/models/qwen3p8-27b"
    tokenizer_model: str = "Qwen/Qwen3.8-27B"
    renderer_name: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    # Fireworks configuration (credentials via FIREWORKS_API_KEY)
    training_base_url: str = "https://api.fireworks.ai/training/v1/serverless"
    init_from_checkpoint: str | None = None

    learning_rate: float = 1e-5
    lr_scheduler: LRSchedulerSpec = field(default_factory=default_constant_schedule)
    kl_beta: float = 0.0
    completions_per_prompt: int = 8
    prompt_groups_per_step: int = 32
    pipeline_chunks_per_step: int = 2
    max_completion_tokens: int = 16384
    max_seq_len: int = 524288
    temperature: float = 1.0
    epochs: int = 1
    shuffle: bool = False
    seed: int = 0
    # Taskset selection happens before the cursor expands tasks across epochs.
    max_tasks: int = 320
    # Maximum additional optimizer updates in this invocation, including on resume.
    max_steps: int | None = None
    lora_rank: int = 32

    adam_beta2: float = DEFAULT_ADAM["beta2"]
    adam_epsilon: float = DEFAULT_ADAM["eps"]
    weight_decay: float = DEFAULT_ADAM["weight_decay"]

    max_head_offpolicy_versions: int = 0
    max_concurrency_rollout_sample: int | None = 32
    min_group_size: int | None = None
    max_incomplete_group_retries: int = 2

    router_replay: bool = True
    router_replay_completion_only: bool = True
    grad_accumulation_normalization: str | None = "num_loss_tokens"
    grad_clip_norm: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    tis: TISConfig = field(default_factory=TISConfig)
    anchor_logp: Literal["old_policy", "rollout"] = "old_policy"
    sample_timeout: float = 2400.0
    snapshot_prefix: str = "ragent-async-rl"
    bridge_host: str = "127.0.0.1"
    bridge_port: int = 0
    eval_tasks: int = 1
    eval_rollouts: int = 2
    log_path: str | None = "outputs"
    metrics_file: str | None = None
    run_name: str | None = None
    step_offset: int = 0
    resolved_tasks_offset: int = 0
    dcp_save_interval: int = 10
    save_final_checkpoint: bool = True
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project=None))

    def __post_init__(self) -> None:
        self.run_name = self.run_name or self.wandb.run_name or _generate_run_name()
        self.wandb = replace(self.wandb, run_name=self.run_name)


@dataclass
class _Snapshot:
    client: Any
    version: int
    leases: int = 0
    requests: int = 0
    retired: bool = False
    closing: bool = False


@dataclass(frozen=True)
class SamplerLease:
    """An episode's immutable snapshot binding, released by its context manager."""

    _owner: Any
    _snapshot: _Snapshot

    @property
    def model(self) -> str:
        return str(self._snapshot.client.deployment_sampler.model)

    @property
    def version(self) -> int:
        return self._snapshot.version

    async def sample_with_prompt_tokens(self, *args: Any, **kwargs: Any) -> Any:
        return await self._owner._sample(self._snapshot, *args, **kwargs)


class ServerlessSampler:
    """Publish snapshots without retiring clients still leased by episodes.

    State changes run without suspension on the owning event loop. Publication
    and the coordinator's synchronous publish therefore form one critical section.
    """

    def __init__(self, client: Any, version: int = 0) -> None:
        self._current = _Snapshot(client, version)
        self._snapshots = {id(client): self._current}
        self._retirement_tasks: set[asyncio.Task] = set()
        self._retirement_errors: list[BaseException] = []
        self._changed = asyncio.Event()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def model(self) -> str:
        return str(self._require_client().deployment_sampler.model)

    @property
    def base_url(self) -> str:
        return str(self._require_client().deployment_sampler.base_url)

    @property
    def tokenizer(self) -> Any:
        return self._require_client().deployment_sampler.tokenizer

    @asynccontextmanager
    async def lease(self):
        self._require_client()
        snapshot = self._current
        snapshot.leases += 1
        try:
            yield SamplerLease(self, snapshot)
        finally:
            snapshot.leases -= 1
            self._retire_if_idle(snapshot)
            self._changed.set()

    async def sample_with_prompt_tokens(self, *args: Any, **kwargs: Any) -> Any:
        self._require_client()
        return await self._sample(self._current, *args, **kwargs)

    async def _sample(self, snapshot: _Snapshot, *args: Any, **kwargs: Any) -> Any:
        if self._closed or snapshot.closing:
            raise RuntimeError("serverless sampler is closed")
        snapshot.requests += 1
        try:
            return await snapshot.client.deployment_sampler.sample_with_prompt_tokens(
                *args,
                **kwargs,
            )
        finally:
            snapshot.requests -= 1
            self._retire_if_idle(snapshot)
            self._changed.set()

    async def replace(self, client: Any, *, version: int | None = None) -> None:
        """Publish without any await between the pointer change and return."""
        self._require_client()
        old = self._current
        if id(client) in self._snapshots:
            raise ValueError("A sampler replacement must be a new client")
        next_version = old.version + 1 if version is None else version
        if next_version <= old.version:
            raise ValueError("Sampler policy versions must increase")
        self._current = _Snapshot(client, next_version)
        self._snapshots[id(client)] = self._current
        old.retired = True
        self._retire_if_idle(old)

    def _retire_if_idle(self, snapshot: _Snapshot) -> None:
        if (
            snapshot.retired
            and not snapshot.closing
            and not (snapshot.leases or snapshot.requests)
        ):
            snapshot.closing = True
            task = asyncio.create_task(self._close_snapshot(snapshot))
            self._retirement_tasks.add(task)
            task.add_done_callback(self._retired)

    def _retired(self, task: asyncio.Task) -> None:
        self._retirement_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self._retirement_errors.append(task.exception())

    async def _close_snapshot(self, snapshot: _Snapshot) -> None:
        try:
            # We use DeploymentSampler directly, so its HTTP client belongs to
            # this loop (not FiretitanSamplingClient's optional background loop).
            # The pinned SDK only exposes synchronous close; drain its async
            # transport here before that close can move it onto another loop.
            transport = getattr(
                snapshot.client.deployment_sampler, "_async_client", None
            )
            if transport is not None:
                await transport.aclose()
            snapshot.client.close()
        finally:
            self._snapshots.pop(id(snapshot.client), None)
            self._changed.set()

    async def aclose(self) -> None:
        self._closed = True
        for snapshot in list(self._snapshots.values()):
            snapshot.retired = True
            self._retire_if_idle(snapshot)
        while self._snapshots:
            self._changed.clear()
            if self._retirement_tasks:
                await asyncio.gather(*self._retirement_tasks, return_exceptions=True)
            else:
                await self._changed.wait()
        if self._retirement_errors:
            raise self._retirement_errors[0]

    def _require_client(self) -> Any:
        if self._closed:
            raise RuntimeError("serverless sampler is closed")
        return self._current.client


def _validate_config(cfg: Config) -> None:
    validate_grpo_config(
        kl_beta=cfg.kl_beta,
        eps_clip=cfg.eps_clip,
        eps_clip_high=cfg.eps_clip_high,
        anchor_logp=cfg.anchor_logp,
    )
    if cfg.kl_beta != 0:
        raise ValueError(
            "serverless async RL does not provision a reference model; "
            "kl_beta must be 0"
        )
    positive = {
        "completions_per_prompt": cfg.completions_per_prompt,
        "prompt_groups_per_step": cfg.prompt_groups_per_step,
        "pipeline_chunks_per_step": cfg.pipeline_chunks_per_step,
        "max_completion_tokens": cfg.max_completion_tokens,
        "max_seq_len": cfg.max_seq_len,
        "max_tasks": cfg.max_tasks,
        "eval_tasks": cfg.eval_tasks,
        "eval_rollouts": cfg.eval_rollouts,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
    if cfg.max_steps is not None and cfg.max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if cfg.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if not math.isfinite(cfg.temperature) or cfg.temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(cfg.sample_timeout) or cfg.sample_timeout <= 0:
        raise ValueError("sample_timeout must be finite and positive")
    if cfg.completions_per_prompt < 2:
        raise ValueError("GRPO requires completions_per_prompt >= 2")
    if not 2 <= _min_group_size(cfg) <= cfg.completions_per_prompt:
        raise ValueError(
            "min_group_size must be in [2, completions_per_prompt] for GRPO"
        )
    if cfg.max_incomplete_group_retries < 0:
        raise ValueError("max_incomplete_group_retries must be >= 0")
    if cfg.max_head_offpolicy_versions < 0:
        raise ValueError("max_head_offpolicy_versions must be >= 0")
    if cfg.step_offset < 0:
        raise ValueError("step_offset must be >= 0")
    if cfg.resolved_tasks_offset < 0:
        raise ValueError("resolved_tasks_offset must be >= 0")
    if (cfg.step_offset or cfg.resolved_tasks_offset) and not cfg.init_from_checkpoint:
        raise ValueError("serverless resume offsets require init_from_checkpoint")
    minimum_resolved = cfg.step_offset * cfg.prompt_groups_per_step
    if cfg.resolved_tasks_offset < minimum_resolved:
        raise ValueError(
            "resolved_tasks_offset cannot be smaller than the prompt groups "
            "represented by step_offset"
        )
    if (
        cfg.max_concurrency_rollout_sample is not None
        and cfg.max_concurrency_rollout_sample < cfg.completions_per_prompt
    ):
        raise ValueError(
            "max_concurrency_rollout_sample must be at least completions_per_prompt"
        )
    if cfg.max_completion_tokens >= cfg.max_seq_len:
        raise ValueError("max_completion_tokens must be smaller than max_seq_len")
    if cfg.lora_rank < 1:
        raise ValueError("serverless async RL requires lora_rank >= 1")
    if not 0.0 <= cfg.adam_beta2 < 1.0:
        raise ValueError("adam_beta2 must be in [0, 1)")
    if cfg.adam_epsilon <= 0.0:
        raise ValueError("adam_epsilon must be > 0")
    if cfg.weight_decay < 0.0:
        raise ValueError("weight_decay must be >= 0")
    if cfg.dcp_save_interval < 0:
        raise ValueError("dcp_save_interval must be >= 0")
    if cfg.bridge_host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("bridge_host must be loopback")
    if not 0 <= cfg.bridge_port <= 65535:
        raise ValueError("bridge_port must be in [0, 65535]")
    if cfg.chat_template_kwargs is not None and not isinstance(
        cfg.chat_template_kwargs, dict
    ):
        raise ValueError("chat_template_kwargs must be an object")
    if cfg.grad_accumulation_normalization not in {None, "num_loss_tokens"}:
        raise ValueError(
            "grad_accumulation_normalization must be None or 'num_loss_tokens'"
        )


def _min_group_size(cfg: Config) -> int:
    return (
        cfg.completions_per_prompt if cfg.min_group_size is None else cfg.min_group_size
    )


def _select_tasks(env: vf.Env, cfg: Config) -> list[vf.Task]:
    tasks = list(env.taskset.head(cfg.max_tasks))
    if not tasks:
        raise ValueError("Verifiers taskset is empty")
    return tasks


def _prepare_run(cfg: Config, env: vf.Env) -> tuple[list[vf.Task], Any, Any]:
    """Validate native rendering and task selection before remote provisioning."""
    _validate_config(cfg)
    validate_environment(env.config)
    tasks = _select_tasks(env, cfg)
    tokenizer = load_tokenizer(cfg.tokenizer_model)
    renderer_config = config_from_name(cfg.renderer_name) if cfg.renderer_name else None
    renderer = create_renderer(
        tokenizer, renderer_config, chat_template_kwargs=cfg.chat_template_kwargs
    )
    if env.taskset.toolsets(env.config.taskset) and not getattr(
        renderer, "supports_tools", True
    ):
        raise ValueError("The selected renderer does not support the taskset's tools")
    return tasks, tokenizer, renderer_config


def _metrics_file(cfg: Config) -> str | None:
    if cfg.metrics_file is not None:
        return cfg.metrics_file
    if cfg.log_path is None:
        return None
    return str(Path(cfg.log_path).expanduser().resolve() / "metrics.jsonl")


def _task_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    # Cookbook scheduling keeps its row vocabulary; local logs describe tasks.
    return {
        key.replace("rows", "tasks") if key.startswith("producer/") else key: value
        for key, value in metrics.items()
    }


@asynccontextmanager
async def _serving(env: vf.Env, bridge: FireworksGenerateServer):
    async with AsyncExitStack() as stack:
        await bridge.start()
        stack.push_async_callback(bridge.stop)
        await stack.enter_async_context(env.serving())
        yield


async def _record_episode(
    cfg: Config,
    episode: vf.Episode,
    version: int,
    context: dict[str, Any],
    write_lock: asyncio.Lock,
) -> None:
    if cfg.log_path is None:
        return
    record = {
        **episode.to_record(),
        "run_name": cfg.run_name,
        "policy_version": version,
        "rollout_context": context,
    }
    line = json.dumps(record, allow_nan=False) + "\n"

    def append() -> None:
        output = Path(cfg.log_path).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        with (output / "episodes.jsonl").open("a", encoding="utf-8") as target:
            target.write(line)

    async with write_lock:
        await asyncio.to_thread(append)


async def _execute_episode(
    cfg: Config,
    *,
    env: vf.Env,
    task: vf.Task,
    sampler: ServerlessSampler,
    bridge: FireworksGenerateServer,
    renderer: Any,
    write_lock: asyncio.Lock,
    context: dict[str, Any],
) -> RolloutRun | None:
    """Execute exactly one fresh slot against one immutable policy snapshot."""
    async with sampler.lease() as lease:
        rollout_id = bridge.register(lease)
        try:
            extras = {ROLLOUT_ID: rollout_id}
            if cfg.chat_template_kwargs is not None:
                extras["chat_template_kwargs"] = cfg.chat_template_kwargs
            ctx = vf.ModelContext(
                model=lease.model,
                client=bridge.train_client_config(cfg.tokenizer_model, renderer),
                sampling=vf.SamplingConfig(
                    max_tokens=cfg.max_completion_tokens,
                    temperature=cfg.temperature,
                    extra_body=extras,
                ),
            )
            episode = await env.run_slot(env.slots(task, 1)[0], ctx)
            await _record_episode(cfg, episode, lease.version, context, write_lock)
            capture = bridge.captures[rollout_id]
            if capture.fatal_error is not None:
                raise capture.fatal_error
            run = episode_to_rollout_run(episode, capture.turns)
            if run is not None:
                run.metadata.update(
                    run_name=cfg.run_name, policy_version=lease.version, **context
                )
            else:
                logger.warning(
                    "Dropped Verifiers episode %s for task %s", episode.id, task.key
                )
            return run
        finally:
            await bridge.release(rollout_id)


def _make_service(base_url: str) -> FiretitanServiceClient:
    headers = read_api_extra_headers_env()
    return FiretitanServiceClient(
        api_key=os.getenv("FIREWORKS_API_KEY"),
        base_url=base_url,
        default_headers=headers,
    )


def _snapshot_name(prefix: str, suffix: str) -> str:
    """Build a DNS label with room for the SDK's 8-character session suffix."""

    prefix_slug = re.sub(r"[^a-z0-9]+", "-", prefix.lower()).strip("-")
    suffix_slug = re.sub(r"[^a-z0-9]+", "-", suffix.lower()).strip("-")
    candidate = "-".join(part for part in (prefix_slug, suffix_slug) if part)
    if not candidate:
        raise ValueError(
            "snapshot prefix and suffix must contain an alphanumeric character"
        )
    if len(candidate) <= _MAX_SNAPSHOT_NAME_LENGTH:
        return candidate

    digest = hashlib.sha256(candidate.encode()).hexdigest()[:8]
    tail = suffix_slug[-20:]
    head_length = _MAX_SNAPSHOT_NAME_LENGTH - len(digest) - len(tail) - 2
    head = prefix_slug[:head_length].rstrip("-")
    return "-".join(part for part in (head, tail, digest) if part)


def _create_snapshot_sampling_client(
    *,
    service: FiretitanServiceClient,
    training_client: Any,
    tokenizer: Any,
    name: str,
) -> tuple[Any, str]:
    saved = training_client.save_weights_for_sampler(name).result()
    snapshot = getattr(saved, "path", None)
    if not snapshot:
        raise RuntimeError(f"serverless sampler snapshot {name!r} has no path")
    client = service.create_sampling_client(
        model_path=snapshot,
        tokenizer=tokenizer,
    )
    return client, snapshot


def _wandb_config(cfg: Config, *, mode: str) -> dict[str, Any]:
    return {
        "run_name": cfg.run_name,
        "mode": mode,
        "algorithm": "grpo",
        "trainer_loss": "client",
        "base_model": cfg.base_model,
        "tokenizer_model": cfg.tokenizer_model,
        "lora_rank": cfg.lora_rank,
        "learning_rate": cfg.learning_rate,
        "adam_beta2": cfg.adam_beta2,
        "adam_epsilon": cfg.adam_epsilon,
        "weight_decay": cfg.weight_decay,
        "completions_per_prompt": cfg.completions_per_prompt,
        "prompt_groups_per_step": cfg.prompt_groups_per_step,
        "pipeline_chunks_per_step": cfg.pipeline_chunks_per_step,
        "max_tasks": cfg.max_tasks,
        "max_steps": cfg.max_steps,
        "max_seq_len": cfg.max_seq_len,
        "max_completion_tokens": cfg.max_completion_tokens,
        "max_head_offpolicy_versions": cfg.max_head_offpolicy_versions,
        "max_concurrency_rollout_sample": cfg.max_concurrency_rollout_sample,
        "step_offset": cfg.step_offset,
        "resolved_tasks_offset": cfg.resolved_tasks_offset,
        "dcp_save_interval": cfg.dcp_save_interval,
        "grad_accumulation_normalization": cfg.grad_accumulation_normalization,
        "router_replay_completion_only": cfg.router_replay_completion_only,
        "kl_beta": cfg.kl_beta,
        "eps_clip": cfg.eps_clip,
        "eps_clip_high": cfg.eps_clip_high,
        "tis_cap": cfg.tis.cap,
        "tis_level": cfg.tis.level,
        "tis_icepop_threshold": cfg.tis.icepop_threshold,
        "anchor_logp": cfg.anchor_logp,
    }


def _sampling_kwargs(cfg: Config, *, router_replay_enabled: bool) -> dict[str, Any]:
    sample_kwargs: dict[str, Any] = {
        "max_tokens": cfg.max_completion_tokens,
        "temperature": cfg.temperature,
        "top_p": 1.0,
        "top_k": 0,
        "max_seq_len": cfg.max_seq_len,
        "http_timeout": cfg.sample_timeout,
        "logprobs": True,
    }
    if router_replay_enabled:
        sample_kwargs.update(
            include_routing_matrix=True,
            echo=not cfg.router_replay_completion_only,
        )
    return sample_kwargs


def _router_replay_enabled(
    cfg: Config, *, api_key: str, base_url: str, training_client: Any | None = None
) -> bool:
    enabled = resolve_router_replay_enabled(
        requested=cfg.router_replay,
        api_key=api_key,
        base_url=base_url,
        additional_headers=read_api_extra_headers_env(),
        base_model=cfg.base_model,
        training_client=training_client,
    )
    if enabled:
        warn_if_full_sequence_router_replay(cfg.router_replay_completion_only)
    return enabled


def run_evaluation(config: Config, *, env: vf.Env) -> dict[str, Any]:
    """Sample native episodes from an initial/restored policy without optimization."""
    cfg = config
    tasks, tokenizer, renderer = _prepare_run(cfg, env)
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    setup_wandb(
        cfg.wandb,
        _wandb_config(cfg, mode="serverless-sampling-eval"),
        metric_steps=ASYNC_RL_WANDB_METRIC_STEPS,
    )
    service = None
    sampler = None
    try:
        service = _make_service(cfg.training_base_url)
        training_client = service.create_lora_training_client(
            base_model=cfg.base_model, rank=cfg.lora_rank
        )
        if cfg.init_from_checkpoint:
            training_client.load_state_with_optimizer(cfg.init_from_checkpoint).result()
        client, _snapshot = _create_snapshot_sampling_client(
            service=service,
            training_client=training_client,
            tokenizer=tokenizer,
            name=_snapshot_name(cfg.snapshot_prefix, f"eval-step-{cfg.step_offset}"),
        )
        sampler = ServerlessSampler(client, version=cfg.step_offset)
        bridge = FireworksGenerateServer(
            _sampling_kwargs(
                cfg,
                router_replay_enabled=_router_replay_enabled(
                    cfg,
                    api_key=api_key,
                    base_url=base_url,
                    training_client=training_client,
                ),
            ),
            host=cfg.bridge_host,
            port=cfg.bridge_port,
        )

        async def sample() -> dict[str, Any]:
            pending: list[asyncio.Task] = []
            try:
                async with _serving(env, bridge):
                    write_lock = asyncio.Lock()
                    gate = asyncio.Semaphore(
                        cfg.max_concurrency_rollout_sample or cfg.completions_per_prompt
                    )

                    async def draw(task: vf.Task, task_index: int, sample_index: int):
                        async with gate:
                            return await _execute_episode(
                                cfg,
                                env=env,
                                task=task,
                                sampler=sampler,
                                bridge=bridge,
                                renderer=renderer,
                                write_lock=write_lock,
                                context={
                                    "task_index": task_index,
                                    "sample_index": sample_index,
                                },
                            )

                    try:
                        pending = [
                            asyncio.create_task(draw(task, task_index, sample_index))
                            for task_index, task in enumerate(tasks[: cfg.eval_tasks])
                            for sample_index in range(cfg.eval_rollouts)
                        ]
                        runs = await asyncio.gather(*pending)
                    finally:
                        for task in pending:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)
                    accepted = [run for run in runs if run is not None]
                    if not accepted:
                        raise TrainingEvidenceError(
                            "Evaluation produced no trainable episodes"
                        )
                    result = {
                        "run_name": cfg.run_name,
                        "eval/tasks": min(len(tasks), cfg.eval_tasks),
                        "eval/reward": sum(run.segments[0].reward for run in accepted)
                        / len(accepted),
                        "eval/episodes": len(runs),
                        "eval/accepted_episodes": len(accepted),
                        "eval/dropped_episodes": len(runs) - len(accepted),
                        "eval/trainable_tokens": sum(
                            sum(segment.loss_mask)
                            for run in accepted
                            for segment in run.segments
                        ),
                        "eval/model_turns": sum(len(run.segments) for run in accepted),
                        "eval/router_replay_segments": sum(
                            segment.routing_matrices is not None
                            for run in accepted
                            for segment in run.segments
                        ),
                    }
                    log_metrics(
                        {"rollout/step": cfg.step_offset, **result},
                        step=cfg.step_offset,
                        metrics_file=_metrics_file(cfg),
                    )
                    return result
            finally:
                await sampler.aclose()

        return asyncio.run(sample())
    finally:
        try:
            if sampler is not None and not sampler.closed:
                asyncio.run(sampler.aclose())
            if service is not None:
                service.close()
        finally:
            wandb_finish(metrics_file=_metrics_file(cfg))


def _require_aligned_logprobs(
    data: list[tinker.Datum],
    logprobs: list[list[float]],
    *,
    source: str,
) -> None:
    if len(logprobs) != len(data):
        raise ValueError(
            f"serverless client GRPO requires one {source} row per datum; "
            f"got {len(logprobs)} rows for {len(data)} datums"
        )
    for index, (datum, row) in enumerate(zip(data, logprobs, strict=True)):
        expected = datum_target_len(datum)
        if not row or len(row) != expected:
            raise ValueError(
                f"serverless client GRPO {source} row {index} must align with "
                f"target_tokens; got {len(row)} values for {expected} targets"
            )


def _require_inference_metrics(result: Any) -> None:
    metrics = getattr(result, "metrics", None) or {}
    required = (
        "inference_k1",
        "inference_k3",
        "raw_inference_logprob_coverage",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise RuntimeError(
            "serverless client GRPO did not report train/inference metrics: "
            + ", ".join(missing)
        )
    for name in required:
        value = float(metrics[name])
        if not math.isfinite(value):
            raise RuntimeError(
                f"serverless client GRPO metric {name} is not finite: {value}"
            )
    if float(metrics["inference_k3"]) < 0:
        raise RuntimeError("serverless client GRPO inference_k3 must be non-negative")
    if float(metrics["raw_inference_logprob_coverage"]) != 1.0:
        raise RuntimeError(
            "serverless client GRPO raw inference logprob coverage must be 1.0"
        )


def main(
    config: Config, *, env: vf.Env, dynamic_filter_fn: DynamicFilterFn | None = None
) -> dict[str, Any]:
    """Train native Verifiers tasks with Fireworks scheduling and GRPO updates."""
    cfg = config
    tasks, tokenizer, renderer = _prepare_run(cfg, env)

    def _signal_handler(signum, _frame):
        name = signal.Signals(signum).name
        raise TerminatedBySignal(name)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    task_loader = CursorDataLoader[vf.Task](
        tasks,
        start_cursor=cfg.resolved_tasks_offset,
        epochs=cfg.epochs,
        shuffle=cfg.shuffle,
        seed=cfg.seed,
    )
    if cfg.resolved_tasks_offset >= task_loader.total_items:
        raise ValueError("No unresolved tasks remain for training")
    lr_scheduler = normalize_lr_scheduler_spec(cfg.lr_scheduler)
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    setup_wandb(
        cfg.wandb,
        _wandb_config(cfg, mode="serverless-training"),
        metric_steps=ASYNC_RL_WANDB_METRIC_STEPS,
    )
    service: FiretitanServiceClient | None = None
    sampler: ServerlessSampler | None = None
    try:
        service = _make_service(config.training_base_url)
        training_client = service.create_lora_training_client(
            base_model=cfg.base_model,
            rank=cfg.lora_rank,
        )
        if cfg.init_from_checkpoint:
            training_client.load_state_with_optimizer(cfg.init_from_checkpoint).result()
        initial_client, initial_snapshot = _create_snapshot_sampling_client(
            service=service,
            training_client=training_client,
            tokenizer=tokenizer,
            name=_snapshot_name(cfg.snapshot_prefix, f"step-{cfg.step_offset}"),
        )
        sampler = ServerlessSampler(initial_client, version=cfg.step_offset)
        router_replay_enabled = _router_replay_enabled(
            cfg,
            api_key=api_key,
            base_url=base_url,
            training_client=training_client,
        )
        bridge = FireworksGenerateServer(
            _sampling_kwargs(cfg, router_replay_enabled=router_replay_enabled),
            host=cfg.bridge_host,
            port=cfg.bridge_port,
        )
        remaining_tasks = task_loader.total_items - cfg.resolved_tasks_offset
        remaining_steps = math.ceil(remaining_tasks / cfg.prompt_groups_per_step)
        if cfg.max_steps is not None:
            remaining_steps = min(remaining_steps, cfg.max_steps)
        total_steps = cfg.step_offset + remaining_steps

        def make_task_requests(write_lock: asyncio.Lock):
            tasks_per_epoch = len(tasks)
            for item in task_loader:
                index = item.index
                epoch = index // tasks_per_epoch
                task_index = index % tasks_per_epoch

                def run_one_episode(
                    sample_index: int,
                    task=item.value,
                    cursor_index=index,
                    task_index=task_index,
                    epoch=epoch,
                ):
                    return _execute_episode(
                        cfg,
                        env=env,
                        task=task,
                        sampler=sampler,
                        bridge=bridge,
                        renderer=renderer,
                        write_lock=write_lock,
                        context={
                            "cursor_index": cursor_index,
                            "task_index": task_index,
                            "epoch": epoch,
                            "rollout_idx": sample_index,
                            "sample_index": sample_index,
                            "end_of_epoch": task_index == tasks_per_epoch - 1,
                        },
                    )

                yield RolloutRow(
                    row_id=index,
                    run_factory=run_one_episode,
                    row_meta={"task_key": item.value.key},
                    on_resolved=lambda _reason, index=index: task_loader.mark_resolved(
                        index
                    ),
                )

        def train_chunk(chunk: TrainingChunk) -> dict[str, Any]:
            groups = list(chunk.groups)
            (
                data,
                advantages,
                ref_logprobs,
                prompt_lens,
                rollout_logprobs,
                raw_inference_logprobs,
            ) = combine_prompt_groups(groups, include_raw=True)
            _require_aligned_logprobs(
                data,
                rollout_logprobs,
                source="rollout behavior logprob",
            )
            _require_aligned_logprobs(
                data,
                raw_inference_logprobs,
                source="raw inference logprob",
            )
            precomputed_forward = None
            if cfg.anchor_logp == "old_policy":
                with elapsed_timer("old_policy_forward"):
                    old_policy_result = training_client.forward(
                        data,
                        "cross_entropy",
                    ).result()
                old_policy_logprobs = [
                    output["logprobs"].data
                    for output in old_policy_result.loss_fn_outputs
                ]
                _require_aligned_logprobs(
                    data,
                    old_policy_logprobs,
                    source="old-policy logprob",
                )
                precomputed_forward = old_policy_result
            else:
                old_policy_logprobs = rollout_logprobs
            with elapsed_timer("fwd_bwd"):
                result = training_client.forward_backward_custom(
                    data,
                    make_grpo_loss_fn(
                        advantages=advantages,
                        ref_logprobs=ref_logprobs,
                        prompt_len=prompt_lens,
                        inf_logprobs=rollout_logprobs,
                        old_policy_logprobs=old_policy_logprobs,
                        kl_beta=cfg.kl_beta,
                        eps_clip=cfg.eps_clip,
                        eps_clip_high=cfg.eps_clip_high,
                        tis_config=cfg.tis,
                        raw_inf_logprobs=raw_inference_logprobs,
                    ),
                    precomputed_forward=precomputed_forward,
                ).result()
                result.metrics["custom_forward_reused"] = float(
                    precomputed_forward is not None
                )
            _require_inference_metrics(result)
            return {"prompt_groups": groups, "fwd_bwd_result": result}

        def optimizer_step(step: int) -> dict[str, Any]:
            learning_rate = compute_lr(
                lr_scheduler,
                step=step,
                base_lr=cfg.learning_rate,
                total_steps=total_steps,
            )
            adam_kwargs = dict(DEFAULT_ADAM)
            adam_kwargs["grad_clip_norm"] = cfg.grad_clip_norm
            adam_kwargs["beta2"] = cfg.adam_beta2
            adam_kwargs["eps"] = cfg.adam_epsilon
            adam_kwargs["weight_decay"] = cfg.weight_decay
            params = tinker.AdamParams(learning_rate=learning_rate, **adam_kwargs)
            with elapsed_timer("optim_step"):
                result = training_client.optim_step(
                    params,
                    grad_accumulation_normalization=(
                        cfg.grad_accumulation_normalization
                    ),
                    emit_grad_norm_metrics=True,
                ).result()
            return {"result": result, "learning_rate": learning_rate}

        def create_sampler_version(step: int) -> tuple[Any, str]:
            return _create_snapshot_sampling_client(
                service=service,
                training_client=training_client,
                tokenizer=tokenizer,
                name=_snapshot_name(cfg.snapshot_prefix, f"step-{step}"),
            )

        def save_training_state(step: int) -> str:
            saved = training_client.save_state(
                _snapshot_name(cfg.snapshot_prefix, f"state-step-{step}")
            ).result()
            path = getattr(saved, "path", None)
            if not path:
                raise RuntimeError(
                    f"serverless DCP checkpoint at step {step} has no path"
                )
            logger.info("[step %d] saved serverless DCP checkpoint: %s", step, path)
            return str(path)

        async def run_training() -> tuple[int, dict[str, Any], str, list[str]]:
            try:
                async with _serving(env, bridge):
                    return await train_serving()
            finally:
                await sampler.aclose()

        async def train_serving() -> tuple[int, dict[str, Any], str, list[str]]:
            periodic_checkpoints: list[str] = []
            telemetry = AsyncRLTelemetry(
                producer_metrics_fn=lambda metrics: log_metrics(
                    _task_metrics(metrics),
                    step=int(metrics["producer/event"]),
                    metrics_file=_metrics_file(cfg),
                ),
                step_metrics_fn=lambda metrics, step: log_metrics(
                    _task_metrics(metrics),
                    step=step,
                    metrics_file=_metrics_file(cfg),
                ),
            )
            coordinator = AsyncRLCoordinator(
                rows=make_task_requests(asyncio.Lock()),
                completions_per_prompt=cfg.completions_per_prompt,
                prompt_groups_per_step=cfg.prompt_groups_per_step,
                training_chunks_per_step=cfg.pipeline_chunks_per_step,
                max_head_off_policy_versions=cfg.max_head_offpolicy_versions,
                max_concurrent_rollouts=cfg.max_concurrency_rollout_sample,
                router_replay_completion_only=(cfg.router_replay_completion_only),
                min_group_size=_min_group_size(cfg),
                max_incomplete_group_retries=(cfg.max_incomplete_group_retries),
                dynamic_filter_fn=dynamic_filter_fn,
                global_step=cfg.step_offset,
                resolved_rows_offset=cfg.resolved_tasks_offset,
                resolved_rows_fn=lambda: task_loader.data_consumed,
            )
            latest_snapshot = initial_snapshot
            async with coordinator:
                telemetry.start(coordinator.snapshot)
                try:
                    while (
                        cfg.max_steps is None
                        or coordinator.global_step - cfg.step_offset < cfg.max_steps
                    ) and (batch := await coordinator.next_batch()) is not None:
                        chunk_outputs: list[dict[str, Any]] = []
                        async for chunk in batch.chunks():
                            coordinator.raise_if_failed(batch)
                            chunk_outputs.append(
                                await coordinator.run_blocking(
                                    "train_chunk",
                                    train_chunk,
                                    chunk,
                                    optimizer_batch=batch,
                                )
                            )
                        coordinator.raise_if_failed(batch)
                        optimizer = await coordinator.run_blocking(
                            "optimizer",
                            optimizer_step,
                            batch.batch_id,
                            optimizer_batch=batch,
                        )
                        with wall_timer(
                            "weight_update",
                            attributes={"step": batch.batch_id},
                        ) as update_span:
                            (
                                next_client,
                                latest_snapshot,
                            ) = await coordinator.run_blocking(
                                "weight_sync",
                                create_sampler_version,
                                batch.batch_id,
                                optimizer_batch=batch,
                            )
                            await sampler.replace(next_client, version=batch.batch_id)
                        published = coordinator.publish(batch)
                        telemetry.finish_step(
                            batch=batch,
                            trained_against_version=(published.trained_against_version),
                            prompt_groups=[
                                group
                                for output in chunk_outputs
                                for group in output["prompt_groups"]
                            ],
                            fwd_bwd_results=[
                                output["fwd_bwd_result"] for output in chunk_outputs
                            ],
                            optim_result=optimizer["result"],
                            timing_metrics=flush_timing(),
                            step_time=published.step_time,
                            weight_update_time=update_span.elapsed,
                            learning_rate=optimizer["learning_rate"],
                        )
                        reached_limit = (
                            cfg.max_steps is not None
                            and batch.batch_id - cfg.step_offset >= cfg.max_steps
                        )
                        if reached_limit:
                            # Close before another await lets publication wake
                            # the producer and launch a fresh cohort we won't train.
                            await coordinator.aclose()
                        interval = cfg.dcp_save_interval
                        completed_steps = batch.batch_id - cfg.step_offset
                        if (
                            interval > 0
                            and completed_steps > 0
                            and completed_steps % interval == 0
                        ):
                            with wall_timer(
                                "checkpoint",
                                category="checkpoint",
                                attributes={"step": batch.batch_id},
                            ) as checkpoint_span:
                                if reached_limit:
                                    checkpoint = await asyncio.to_thread(
                                        save_training_state, batch.batch_id
                                    )
                                else:
                                    checkpoint = await coordinator.run_blocking(
                                        "checkpoint",
                                        save_training_state,
                                        batch.batch_id,
                                    )
                            periodic_checkpoints.append(checkpoint)
                            log_metrics(
                                {
                                    "rollout/step": batch.batch_id,
                                    "checkpoint/wall_time": checkpoint_span.elapsed,
                                },
                                step=batch.batch_id,
                                metrics_file=_metrics_file(cfg),
                            )
                finally:
                    await telemetry.aclose()
            return (
                coordinator.global_step,
                telemetry.final_stats(),
                latest_snapshot,
                periodic_checkpoints,
            )

        steps, final_stats, final_snapshot, periodic_checkpoints = asyncio.run(
            run_training()
        )
        resolved_tasks = int(final_stats["resolved_rows"])
        final_checkpoint = None
        if cfg.save_final_checkpoint and (
            steps > cfg.step_offset or resolved_tasks > cfg.resolved_tasks_offset
        ):
            if (
                periodic_checkpoints
                and cfg.dcp_save_interval > 0
                and (steps - cfg.step_offset) % cfg.dcp_save_interval == 0
            ):
                final_checkpoint = periodic_checkpoints[-1]
            else:
                final_checkpoint = save_training_state(steps)
        session = getattr(service, "training_session_name", None) or getattr(
            service,
            "training_session_id",
            None,
        )
        return {
            "run_name": cfg.run_name,
            "steps": steps,
            "accepted_groups": (
                cfg.step_offset * cfg.prompt_groups_per_step
                + int(final_stats["total_accepted"])
            ),
            "resolved_tasks": resolved_tasks,
            "training_session": session,
            "training_run": getattr(training_client, "run_name", None)
            or getattr(training_client, "run_id", None),
            "final_sampler_snapshot": final_snapshot,
            "periodic_training_checkpoints": periodic_checkpoints,
            "final_training_checkpoint": final_checkpoint,
        }
    finally:
        try:
            if sampler is not None and not sampler.closed:
                asyncio.run(sampler.aclose())
            if service is not None:
                service.close()
        finally:
            wandb_finish(metrics_file=_metrics_file(cfg))
