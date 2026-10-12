# Fireworks RL with Verifiers

Train a LoRA policy with GRPO on native Verifiers tasks. Verifiers loads the
taskset, renders conversations, runs the agent and retrieval tools, and scores
episodes. The Fireworks async loop groups independent episodes for each task,
applies training updates, publishes sampler snapshots, and saves checkpoints.
Each episode keeps the same policy snapshot across all of its turns.

Requires Python 3.12 and `uv`. From the repository root:

```bash
cd train
uv sync
cp -n .env-template .env
```

Fill `.env` with `FIREWORKS_API_KEY`, `TURBOPUFFER_API_KEY`, and `PRIME_API_KEY`
for the supplied environment's judge. Set `WANDB_ENTITY` to enable W&B logging;
add `WANDB_API_KEY` for online logging. Existing environment variables take
precedence over `.env`.

Edit the supplied training and environment TOML files to choose the model,
dataset, retrieval namespace, and judge. The environment example points to a
`diegi97/ragent-rubrics`. Defaults use Qwen3.8-27B and 16,384 completion tokens
per model turn.

Run training:

```bash
uv run python main.py @ configs/train.toml
```

Evaluate the initial or restored policy on the selected taskset:

```bash
uv run python main.py @ configs/train.toml mode=eval eval_tasks=2
```

Limit training to one additional optimizer update:

```bash
uv run python main.py @ configs/train.toml max_steps=1
```

Inspect settings and their sources without starting a run:

```bash
uv run python main.py @ configs/train.toml --help
```

CLI `key=value` arguments override the training TOML.

Episode records and metrics are written to `episodes.jsonl` and `metrics.jsonl`
under `log_path` (`outputs/` by default). Checkpoints are stored in Fireworks,
with their paths logged by the launcher. To resume, configure
`init_from_checkpoint`, `step_offset`, and `resolved_tasks_offset`, retaining
the same task selection, epochs, shuffle setting, and seed.
