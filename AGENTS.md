# Repository Guidelines

## Project Structure & Module Organization
- `ragent_core/`: Core Python package (config, prompts, rewards, data_sources, `bm25_client.py`). Env vars loaded from `.env` via `ragent_core/ragent_core/config`.
- `environments/`: Verifiers evaluation/training environments (e.g., `bm25/` with `bm25.py`, its own `pyproject.toml` and `.venv`). Wrapper script: `environments/run_eval.sh`.
- `data/`: Local artifacts (e.g., BM25 indexes under `data/<dataset>/bm25s_corpus_index/`).
- `train.py`: Tyro-driven entrypoint that wires Verifiers’ `GRPOTrainer`.

## Build, Test, and Development Commands
- Setup (Python 3.11+): `cd ragent_core && uv sync` and `cd environments/bm25 && uv sync`.
- Run eval (quick start): `bash environments/run_eval.sh gpt-4.1-mini bm25`.
- Run eval (manual): `cd environments && uv run vf-eval bm25 -m gpt-4.1-mini --save-dataset`.
- Train (example): `uv run python train.py --env-id bm25 --hf-dataset nampdn-ai/devdocs.io --model-name Qwen/Qwen2.5-0.5B --max-steps 100`.

## Coding Style & Naming Conventions
- Indentation: 4 spaces. Use type hints where practical.
- Naming: modules/functions/vars `snake_case`; classes `CapWords`.
- Logging: use configured logger (see `ragent_core/ragent_core/config/logging.py`), avoid `print`.
- Formatting: `uv run black ragent_core environments/bm25` and `uv run isort ragent_core environments/bm25`.

## Testing Guidelines
- Framework: no unit tests yet; validate via environment evals.
- Smoke tests: `uv run vf-eval bm25 -n 10 -r 1 -t 512 --save-dataset`.
- Artifacts: inspect `results/<model>_<env>/` and, if needed, open `index.html` to visualize trajectories.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject (<=72 chars). Optional scope prefixes like `core:`, `env/bm25:`, `docs:`. Example: `env/bm25: integrate judge reward backoff`.
- PRs: clear description, rationale, and reproduction (commands/paths). Link issues. Include screenshots or a sample `results/<model>_<env>/` when behavior changes.

## Security & Configuration Tips
- Do not commit secrets. Use `.env` files (see `environments/bm25/.env-template`).
- Common vars: `OPENROUTER_API_KEY`, `HF_TOKEN`, `WANDB_PROJECT`, `WANDB_API_KEY`, `JUDGE_MODEL`, `OPENROUTER_URL` (base URL). Set them in the relevant `.env`.
