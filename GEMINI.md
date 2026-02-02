# RAGent Context for Gemini

## Project Overview
**RAGent** is a Python-based research project designed to train intelligent agents for multi-hop Question Answering (QA) tasks. It leverages Reinforcement Learning from Verifiable Rewards (RLVR) to teach agents how to navigate and extract information from diverse knowledge bases (e.g., lexical search, vector databases).

The core philosophy is to use **modular environments** where agents interact with specific search engines and tools. The project currently focuses on lightweight lexical search (BM25s) over developer documentation but is architected to support complex setups like Elasticsearch or hybrid search.

## Key Technologies
-   **Language:** Python 3.11+
-   **Package Manager:** `uv` (for fast dependency management)
-   **CLI Parsing:** `tyro`
-   **Agent Framework:** `verifiers` (for RL environments, rubrics, and evaluation)
-   **Retrieval:** `bm25s` (efficient lexical search)
-   **LLM Interaction:** `openai` (and compatible endpoints via OpenRouter)

## Directory Structure
*   **`ragent_core/`**: The heart of the library.
    *   `config/`: Configuration for endpoints, logging, and training.
    *   `pipelines/`: Logic for data generation (e.g., `explorer_agent`, `atomic`).
    *   `prompts/`: System prompts for agents and judges.
    *   `retrievers/`: Implementations of retrieval logic.
    *   `bm25_client.py`: Client wrapper for BM25s interactions.
*   **`environments/`**: standalone environment definitions for the `verifiers` library.
    *   `bm25/`: The primary development environment. Contains its own `pyproject.toml` and dependencies.
*   **`data/`**: Storage for local datasets and BM25 indexes.
*   **`train.py`**: The main training entry point using `GRPOTrainer`.
*   **`run_eval.sh`**: Helper script to run evaluations.

## Development Workflows

### 1. Environment Setup
The project uses `uv` workspaces. You often need to sync dependencies in both the root and specific environment directories.
```bash
# Root setup
uv sync

# Environment setup (e.g., for BM25)
cd environments/bm25 && uv sync
```

### 2. Running Evaluations
Use the provided shell script to run evaluations using the `verifiers` framework (`vf-eval`).
```bash
# Usage: ./run_eval.sh <model_name> <env_name> [qa_dataset_path] [bm25_dataset_path]
./run_eval.sh gemini-2.5-flash-lite bm25
```
*   **Model:** Defaults to `gemini-2.5-flash-lite`.
*   **Environment:** Defaults to `bm25` (maps to `environments/bm25/bm25.py`).
*   **Datasets:** Can be HuggingFace IDs or local paths.

### 3. Training
Training is driven by `train.py` which uses `tyro` for CLI arguments.
```bash
uv run python train.py \
    --env-id bm25 \
    --hf-dataset nampdn-ai/devdocs.io \
    --model-name Qwen/Qwen2.5-0.5B \
    --max-steps 100
```

### 4. Code Formatting
The project enforces strict formatting using `black` and `isort`.
```bash
make format_code
# OR manually
uv run black ragent_core environments/bm25
uv run isort ragent_core environments/bm25
```

## Critical Concepts
*   **Environments**: Defined in `environments/<name>/<name>.py`. They must expose a `load_environment` function that returns a `vf.Environment` (usually `vf.ToolEnv`).
*   **Rubrics**: Used by `verifiers` to score agent performance. Defined in `load_environment` (e.g., `judge_reward`, `format_reward`).
*   **Data Sources**: The project prioritizes non-web sources (email archives, docs).
*   **Local vs. HF**: Datasets and indexes can often be loaded from local paths for faster iteration during development.

## Common Issues / Tips
*   **Configuration**: Check `.env` files for missing keys (`OPENROUTER_API_KEY`, `HF_TOKEN`, etc.).
*   **Paths**: When running commands from the root, ensure relative paths to `environments` are correct.
*   **`run_eval.sh`**: This script automatically loads variables from `.env` before running `vf-eval`.
