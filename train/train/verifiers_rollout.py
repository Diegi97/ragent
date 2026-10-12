"""Convert committed Verifiers model turns into Fireworks training segments."""

import math

import verifiers.v1 as vf
from training.utils.rl.rollout.renderer import sampled_completion_to_rollout_run
from training.utils.rl.rollout.types import RolloutRun, validate_rollout_run_routing

from train.fireworks_bridge import (
    CapturedTurn,
    TrainingEvidenceError,
    completion_logprobs,
)


def _check_failure(episode: vf.Episode) -> bool:
    if episode.ok:
        return False
    errors = [*episode.errors, *(e for t in episode.traces for e in t.errors)]
    if not errors:
        return True
    # Errors are episode data in Verifiers. Re-surface deterministic failures so
    # the async producer cannot quietly exhaust the dataset on a broken setup.
    for error in errors:
        if (
            error.type == "TaskError"
            or (
                error.status_code is not None
                and 400 <= error.status_code < 500
                and error.status_code not in {408, 429}
            )
            or (
                error.status_code is None
                and error.type
                not in {
                    "SandboxError",
                    "InterceptionError",
                    "TunnelError",
                    "TimeoutError",
                }
                and "timed out" not in error.message.lower()
            )
        ):
            raise TrainingEvidenceError(
                f"Episode {episode.id} failed: {error.type}: {error.message}"
            )
    return True


def _path_tokens(trace: vf.Trace, node_id: int) -> list[int]:
    path: list[int] = []
    seen: set[int] = set()
    current: int | None = node_id
    while current is not None:
        if current in seen or not 0 <= current < len(trace.nodes):
            raise TrainingEvidenceError("Invalid Verifiers token ancestry")
        seen.add(current)
        path.append(current)
        current = trace.nodes[current].parent
    return [token for index in reversed(path) for token in trace.nodes[index].token_ids]


def episode_to_rollout_run(
    episode: vf.Episode, captured_turns: list[CapturedTurn]
) -> RolloutRun | None:
    if _check_failure(episode):
        return None
    traces = [trace for trace in episode.traces if trace.agent.trainable]
    if len(traces) != 1:
        raise TrainingEvidenceError("Expected exactly one trainable Verifiers trace")
    trace = traces[0]
    if (
        not trace.ok
        or not trace.rewards
        or any(
            reward is None or not math.isfinite(reward.value)
            for reward in trace.rewards.values()
        )
        or not math.isfinite(trace.reward)
    ):
        raise TrainingEvidenceError("Episode has missing or nonfinite required scoring")
    segments = []
    committed: set[int] = set()
    used: set[int] = set()
    for call in trace.calls:
        if call.node is None or call.error is not None:
            continue
        if call.node in committed or not 0 <= call.node < len(trace.nodes):
            raise TrainingEvidenceError("Invalid or repeated committed model call")
        committed.add(call.node)
        node = trace.nodes[call.node]
        count = sum(node.mask)
        if not count:
            continue
        if (
            not node.sampled
            or len(node.mask) != len(node.token_ids)
            or node.mask != [False] * (len(node.token_ids) - count) + [True] * count
            or len(node.logprobs) != count
        ):
            raise TrainingEvidenceError("Malformed committed assistant token span")
        full_tokens = _path_tokens(trace, call.node)
        matches = [
            index
            for index, turn in enumerate(captured_turns)
            if index not in used
            and turn.trace_id == trace.id
            and turn.completion.full_tokens == full_tokens
            and turn.completion_ids == node.token_ids[-count:]
            and completion_logprobs(turn.completion, "sampling_logprobs")
            == node.logprobs
        ]
        if len(matches) != 1:
            raise TrainingEvidenceError(
                f"Trace {trace.id} node {call.node}: expected one captured turn, "
                f"found {len(matches)}"
            )
        index = matches[0]
        used.add(index)
        completion = captured_turns[index].completion
        completion_logprobs(completion, "inference_logprobs")
        run = sampled_completion_to_rollout_run(completion, reward=trace.reward)
        if run is None:
            raise TrainingEvidenceError("Committed turn has no valid training evidence")
        segments.extend(run.segments)
    sampled = {i for i, n in enumerate(trace.nodes) if n.sampled and any(n.mask)}
    if not sampled <= committed:
        raise TrainingEvidenceError("A sampled node has no committed model call")
    if not segments:
        return None
    run = RolloutRun(
        segments=segments,
        run_id=episode.id,
        metadata={
            "episode_id": episode.id,
            "task_key": episode.task.key,
            "trace_id": trace.id,
            "reward_components": {
                name: reward.model_dump()
                for name, reward in trace.rewards.items()
                if reward is not None
            },
            "metrics": trace.metrics,
        },
    )
    validate_rollout_run_routing(run)
    return run
