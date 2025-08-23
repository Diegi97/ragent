# Repository Guidelines

## Project Structure & Module Organization
- `ragent/`: Core package.
  - `config/`: Env vars, endpoints, training config (`TrainConfig`).
  - `environments/`: Task definitions (e.g., `bm25s/`).
  - `prompts/`: Agent and judge prompts/parsers.
  - `rewards/`: Reward functions (LLM-judge, format checks).
  - `data/`: Data sources and pipelines.
- `data/`: Local datasets and artifacts.
- `results/`: Eval outputs (e.g., `results/<model>_<env>/trajectories.json`).
- `train.py`: Training entrypoint (Tyro CLI over `TrainConfig`).
- `run_eval.sh`: Wrapper to run `vf-eval` and export trajectories.
- `index.html`: Simple visualizer for `results/`.
- `.env-template` → copy to `.env` for local secrets.

## Build, Test, and Development Commands
- `uv run python train.py --help`: Show all training options.
- `uv run python train.py --model-name Qwen/Qwen2.5-0.5B --max-steps 100`: Start a short training run.
- `./run_eval.sh gpt-4.1-mini bm25s`: Run an evaluation and write to `results/`.
- `make format_code` (or `make format_black`, `make format_isort`): Apply Black and isort.

## Coding Style & Naming Conventions
- Python 3.11; prefer type hints and dataclasses for configs.
- Formatting: Black (88 cols) and isort; 4-space indentation.
- Naming: modules/paths `snake_case`; classes `PascalCase`; functions/vars `snake_case`; constants `UPPER_SNAKE_CASE`.
- Organization: config in `ragent/config/`; new environments under `ragent/environments/<name>/` with `load_environment`.

## Testing Guidelines
- No unit-test suite yet. For functional checks, run `./run_eval.sh <model> <env>` and verify `results/<model>_<env>/` artifacts and `trajectories.json`.
- When contributing logic-heavy code, add `pytest` tests under `tests/` (mirror package structure) and provide sample data under `data/` if needed.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (≤72 chars), optional scope (e.g., `env:` `data:` `rewards:`). Group related changes.
- PRs: clear description, linked issues, reproduction steps (commands used), and example output path (e.g., `results/gpt-4.1-mini_bm25s/`). Keep diffs focused and pass formatters.

## Security & Configuration Tips
- Copy `.env-template` to `.env` and set keys used in `ragent/config/` (e.g., `OPENAI_API_KEY`, `HF_TOKEN`, `WANDB_PROJECT`, `JUDGE_MODEL`, `JUDGE_CLIENT`).
- Do not commit secrets; `.env` is gitignored. Prefer `uv run` to ensure isolated, reproducible runs.
