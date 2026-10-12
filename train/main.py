"""Train a Fireworks serverless policy on native Verifiers tasks.

From train/: uv run python main.py @ configs/train.toml mode=eval
"""

import json
import logging
import os
import sys
import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal

import chz
from chz.blueprint import EntrypointHelpException, exit_on_entrypoint_error
from dotenv import load_dotenv
from train.environment import load_environment
from training.utils import WandBConfig

from train import rl_loop

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Train or evaluate a native Verifiers environment.

    Load a TOML [train] block with: python main.py @ configs/train.toml
    Append key=value arguments to override settings from that file.
    """

    env_config: str
    mode: Literal["train", "eval"] = "train"
    base_model: str = "accounts/fireworks/models/qwen3p8-27b"
    tokenizer_model: str = "Qwen/Qwen3.8-27B"
    renderer_name: str | None = None
    lora_rank: int = 32
    training_base_url: str = "https://api.fireworks.ai/training/v1/serverless"
    init_from_checkpoint: str | None = None

    bridge_host: str = "127.0.0.1"
    bridge_port: int = 0
    completions_per_prompt: int = 8
    prompt_groups_per_step: int = 32
    pipeline_chunks_per_step: int = 2
    min_group_size: int | None = None
    max_incomplete_group_retries: int = 2
    learning_rate: float = 1e-5
    kl_beta: float = 0.0
    max_concurrent_rollouts: int = 32
    max_head_offpolicy_versions: int = 0
    max_steps: int | None = None
    max_tasks: int = 320
    epochs: int = 1
    shuffle: bool = False
    seed: int = 0

    max_completion_tokens: int = 16384
    max_seq_len: int = 524288
    temperature: float = 1.0
    sample_timeout: float = 2400.0
    chat_template_kwargs: dict[str, Any] | str | None = None
    router_replay: bool = True
    router_replay_completion_only: bool = True
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    anchor_logp: Literal["old_policy", "rollout"] = "old_policy"
    grad_clip_norm: float = 0.0

    eval_tasks: int = 1
    eval_rollouts: int = 2
    dcp_save_interval: int = 10
    save_final_checkpoint: bool = True
    step_offset: int = 0
    resolved_tasks_offset: int = 0
    snapshot_prefix: str = "ragent-async-rl"
    log_path: str = "outputs"
    entity: str | None = None
    project: str | None = None
    run_name: str | None = None


def make_loop_config(config: CLIConfig) -> rl_loop.Config:
    kwargs = {
        field.name: getattr(config, field.name)
        for field in fields(rl_loop.Config)
        if hasattr(config, field.name)
    }
    kwargs["chat_template_kwargs"] = (
        json.loads(config.chat_template_kwargs)
        if isinstance(config.chat_template_kwargs, str)
        else config.chat_template_kwargs
    )
    return rl_loop.Config(
        **kwargs,
        max_concurrency_rollout_sample=config.max_concurrent_rollouts,
        wandb=WandBConfig(
            entity=config.entity or os.environ.get("WANDB_ENTITY") or None,
            project=config.project or os.environ.get("WANDB_PROJECT") or None,
            run_name=config.run_name,
        ),
    )


def main(config: CLIConfig) -> dict:
    load_dotenv()
    if config.mode not in {"train", "eval"}:
        raise ValueError("mode must be train or eval")
    loop_config = make_loop_config(config)
    logger.info("Starting %s run: %s", config.mode, loop_config.run_name)
    env = load_environment(
        config.env_config, split="train" if config.mode == "train" else "test"
    )
    if config.mode == "eval":
        result = rl_loop.run_evaluation(loop_config, env=env)
    else:
        result = rl_loop.main(loop_config, env=env)
    logger.info("%s complete: %s", config.mode, result)
    return result


def parse_config(argv: list[str] | None = None) -> CLIConfig:
    """Layer a TOML [train] block under chz's typed CLI overrides."""
    args = list(sys.argv[1:] if argv is None else argv)
    blueprint = chz.Blueprint(CLIConfig)
    if args and args[0] == "@":
        if len(args) < 2:
            raise ValueError("Expected a TOML file after @")
        path = Path(args[1]).expanduser().resolve()
        with path.open("rb") as source:
            document = tomllib.load(source)
        if set(document) != {"train"} or not isinstance(document["train"], dict):
            raise ValueError(f"Expected a single [train] block in {path}")
        values = document["train"]
        # File paths belong to the TOML's directory. CLI paths belong to the cwd.
        for key in ("env_config", "log_path"):
            if isinstance(values.get(key), str):
                values[key] = str(
                    (path.parent / Path(values[key]).expanduser()).resolve()
                )
        blueprint.apply(values, layer_name=str(path))
        args = args[2:]
    return blueprint.make_from_argv(args)


@exit_on_entrypoint_error
def cli() -> None:
    try:
        config = parse_config()
    except EntrypointHelpException as error:
        sys.stdout.write(str(error) + "\n")
        return
    except (TypeError, ValueError, OSError) as error:
        logger.error("%s", error)
        raise SystemExit(2) from None
    main(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
