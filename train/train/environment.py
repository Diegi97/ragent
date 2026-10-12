"""Load native Verifiers configuration and retain its typed tasks."""

import tomllib
from pathlib import Path
from typing import Any

import verifiers.v1 as vf


def _resolve_paths(value: dict[str, Any], directory: Path) -> None:
    for key, item in value.items():
        if isinstance(item, dict):
            _resolve_paths(item, directory)
        elif isinstance(item, str) and key in {"env_file", "system_prompt"}:
            value[key] = str((directory / Path(item).expanduser()).resolve())
        elif isinstance(item, str) and key == "dataset_path":
            path = Path(item).expanduser()
            candidate = directory / path
            # Hub identifiers also contain a slash. Only local-file evidence or an
            # explicit filesystem path makes this a local dataset.
            if (
                path.is_absolute()
                or item.startswith((".", "~"))
                or path.suffix.lower() == ".jsonl"
                or candidate.exists()
            ):
                value[key] = str(candidate.resolve())


def load_environment(path: str | Path, *, split: str | None = None) -> vf.Env:
    """Load the environment, applying the launcher's split before task loading."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as source:
        document = tomllib.load(source)
    raw = document.get("env")
    if not isinstance(raw, dict):
        raise ValueError(f"Expected an [env] block in {config_path}")
    if split is not None:
        taskset = raw.setdefault("taskset", {})
        if not isinstance(taskset, dict):
            raise ValueError(f"Expected an [env.taskset] block in {config_path}")
        taskset["split"] = split
    _resolve_paths(raw, config_path.parent)
    config = vf.resolve_env_config(raw)
    validate_environment(config)
    return vf.load_environment(config)


def validate_environment(config: vf.EnvConfig) -> None:
    """Reject agent settings that bypass the training policy."""
    agents = [v for v in config.__dict__.values() if isinstance(v, vf.AgentConfig)]
    if len(agents) != 1:
        raise ValueError("Training currently requires a single-agent environment")
    if agents[0].model is not None or agents[0].client is not None:
        raise ValueError("Remove env agent model/client overrides to use the RL policy")
    extras = (agents[0].sampling.extra_body or {}) if agents[0].sampling else {}
    if "ragent_rollout_id" in extras:
        raise ValueError("ragent_rollout_id is reserved for the training bridge")
