# RAGent

RAGent is a research project focused on training language models to perform agentic search through reinforcement learning. The goal is to train models that autonomously learn to interact with search systems, whether they are Elasticsearch indexes, vector databases, BM25 engines, file systems, SQL databases, or hybrid combinations of these.

## Project Structure

The project is organized into three main development areas:

### 1. Synthetic Data Generation

A pipeline for generating question-answer pairs of varying complexity given a specific data source. The current approach uses an **explorer agent pipeline** (`ragent_core/pipelines/explorer_agent/`), inspired by the agentic search synthesis methodology described in [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556). The pipeline:

- Extracts key concepts from documents in the corpus
- Assigns breadth and depth parameters to each concept based on occurrence frequency
- Uses specialized agents (breadth, depth, synthesis) to generate progressively complex QA pairs
- Leverages retrieval tools during generation to ground questions in actual document content

A graph-based approach for QA generation, inspired by [WebShaper](https://arxiv.org/abs/2507.15061)'s formalization-driven synthesis framework, is currently under development.

### 2. Environment Development

The environment layer (`environments/`) handles the search system setup, curriculum learning, and reward functions. Each environment provides:

- **Search Engine Layer**: The retrieval backend (BM25, vector search, hybrids, etc)
- **Tool Interface**: Actions available to the agent (`search(query)`, `read(doc_id)`, etc.)
- **Data Source**: The knowledge base (a collection of documents to search over)
- **QA Dataset**: Question-answer pairs used for training and evaluation
- **Reward Functions**: LLM-based judges and other reward signals (e.g., format compliance, tool usage) for evaluating agent trajectories

Current environments:
- `bm25/` — Lightweight lexical search environment using BM25s.

### 3. Training

Training is performed using the [verifiers](https://github.com/PrimeIntellect-ai/verifiers) library and [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) library.

## Data Sources

QA pairs have been generated from the following data sources for now:

- GitLab Handbook
- IETF RFCs
- Python Enhancement Proposals (PEPs)
- DevDocs.io developer documentation
- OWASP Cheat Sheets
- Rust RFCs
- Domain-specific corpora (marine biology, space exploration, mythology) extracted from Wikipedia.

## Current Development

Training is starting with approximately 2,500 QA pairs generated from the data sources listed above using the explorer agent pipeline. The initial focus is on lexical search over diverse technical documentation.
