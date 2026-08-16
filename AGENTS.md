# Repository Guidelines

## Project Structure

- The root `pyproject.toml` and `uv.lock` contain repository-wide development tooling only.
- `ragent_core/` is the shared library. Its package contains configuration, data-source loaders, judges, prompts, retrievers, shared types, and utilities; operational scripts live in `ragent_core/scripts/`.
- `data_pipelines/` is the Prefect and Phoenix project for search-query generation, retrieval evaluation, and deep-search task generation. It depends on `ragent_core` through a local editable source.
- `model_services/` contains the BentoML Harrier embedding and mxbai reranking services.
- `environments/ragent_deep_search/` is the standalone Verifiers evaluation and training environment.
- Runtime indexes live in Turbopuffer and resolve through a logical-namespace catalog. Generated pipeline data and evaluation traces live under `data_pipelines/data/` and `environments/ragent_deep_search/outputs/` respectively.

Each subproject has its own `pyproject.toml`, `.python-version`, `.venv`, and `uv.lock`. Run dependency and application commands from the relevant project directory unless a command explicitly targets the repository root.

## Setup and Common Commands

Python 3.12 or newer and `uv` are required. Use locked installs:

```bash
(cd ragent_core && uv sync --locked)
(cd data_pipelines && uv sync --locked)
(cd model_services && uv sync --locked)
(cd environments/ragent_deep_search && uv sync --locked)
```

Run the search-query pipeline:

```bash
cd data_pipelines
docker compose up -d
uv run generate-search-queries run --table-name <table> --num-queries 10
```

Run the model services from `model_services/` with `./run_harrier.sh` and `./run_mxbai.sh`.

Validate and run the deep-search evaluation:

```bash
cd environments/ragent_deep_search
uv run --env-file .env eval @ evaluation.toml --dry-run
uv run --env-file .env eval @ evaluation.toml
```

There is no root-level training entrypoint. Use `ragent_deep_search` with the training workflow provided by `prime-rl`.

## Formatting and Linting

Use 4-space indentation, practical type hints, `snake_case` for modules/functions/variables, and `CapWords` for classes. Use the configured logger rather than `print` in library and pipeline code.

Formatting, import sorting, and linting use the Ruff version pinned by the root lockfile:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
```

Apply changes with `uv run ruff check --fix .` followed by `uv run ruff format .`.

## Testing and Validation

Run the core, pipeline, and model-service tests from their projects:

```bash
(cd ragent_core && uv run pytest tests)
(cd data_pipelines && uv run pytest)
cd model_services
uv run pytest
```

For pipeline changes, run the relevant CLI with the smallest practical input. For environment changes, first run the evaluation dry-run; a real smoke test requires credentials, a dataset, and ready Turbopuffer corpora. The opt-in core live test requires `RAGENT_RUN_TURBOPUFFER_LIVE=1`. When dependencies change, run `uv lock --check --project <project-path>` for every affected project.

Inspect pipeline artifacts under `data_pipelines/data/` and evaluation `traces.jsonl` files under `environments/ragent_deep_search/outputs/`.

## Commit and Pull Request Guidelines

- Use imperative, concise commit subjects of at most 72 characters. Optional scopes include `core:`, `pipelines:`, `model-services:`, `env/deep-search:`, and `docs:`.
- Pull requests should explain the change and rationale, list exact reproduction or validation commands, and link relevant issues.
- Include a small output or trace example when pipeline or environment behavior changes, but never commit generated corpora, model artifacts, indexes, or credentials.

## Security and Configuration

Never commit secrets. Use uncommitted `.env` files and start from `ragent_core/.env-template`, `data_pipelines/.env-template`, or the configuration example in `environments/ragent_deep_search/README.md`.

Common variables include `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `PRIME_API_KEY`, `HF_TOKEN`, `FIREWORKS_ACCOUNT_ID`, `TURBOPUFFER_API_KEY`, `TURBOPUFFER_REGION`, `TURBOPUFFER_NAMESPACE_PREFIX`, `RAGENT_EMBEDDING_SERVICE_URL`, `RAGENT_RERANKER_SERVICE_URL`, and `RAGENT_RETRIEVER_AUTHKEY`. Keep local service-account files under ignored paths such as `secrets/`.
