# BM25 Search Environment

### Overview
- **Environment ID**: `bm25-search`
- **Short description**: Environment for training agentic multi-hop search using BM25 lexical retrieval
- **Tags**: `bm25`, `agentic-search`, `multi-hop`, `tool-use`, `single-turn`

### Datasets
- **Primary dataset(s)**: [ragent_qa_pairs](https://huggingface.co/datasets/diegi97/ragent_qa_pairs), [ragent_data_sources](https://huggingface.co/datasets/diegi97/ragent_data_sources)
- **Source links**: [GitHub](https://github.com/Diegi97/ragent)
- **Split sizes**: 2,374 train / 229 eval QA pairs across 9 data sources

### Task
- **Type**: Multi-turn tool use
- **Parser**: XML-based tool parser
- **Rubric overview**: LLM judge (0.8) + format compliance (0.2)

---

## About RAGent

RAGent is a research project focused on training language models to perform **agentic search** through reinforcement learning. The goal is to train models that can autonomously perform multi-hop searches across document collections, breaking down complex information needs into sequences of search queries, reading relevant documents, and synthesizing answers.

This environment focuses on **lexical search using BM25**. The simplicity of BM25 makes it easy to deploy for training while still giving models the tools and instructions needed to perform successful multi-hop searches. Models trained here are intended to:

- Formulate effective search queries
- Navigate search results and read relevant documents
- Perform iterative refinement when initial queries don't return useful results
- Synthesize information from multiple documents into coherent answers

## Datasets (Detailed)

### QA Pairs

| Split | Total | gitlab_handbook | common_pile_peps | rust_rfcs | rfc_all | owasp_cheatsheets | nampdn_ai_devdocs_io | diegi97_mythology | diegi97_marine_biology | diegi97_space_exploration |
|-------|-------|-----------------|------------------|-----------|---------|-------------------|----------------------|-------------------|------------------------|---------------------------|
| Train | 2,374 | 361 | 355 | 355 | 318 | 225 | 213 | 183 | 183 | 181 |
| Eval | 229 | 39 | 35 | 34 | 31 | 22 | 18 | 16 | 17 | 17 |

### Data Sources

The document corpora include:

- **GitLab Handbook** - Internal documentation from GitLab
- **IETF RFCs** (`rfc_all`) - Internet Engineering Task Force Request for Comments
- **Python Enhancement Proposals** (`common_pile_peps`) - PEPs from the Python community
- **DevDocs.io** (`nampdn_ai_devdocs_io`) - Developer documentation aggregator
- **OWASP Cheat Sheets** - Security best practices documentation
- **Rust RFCs** - Rust language design proposals
- **Domain-specific corpora** - Wikipedia extractions on marine biology, space exploration, and mythology

### QA Generation Pipeline

QA pairs are generated using the **Explorer Agent Pipeline** (`ragent_core/pipelines/explorer_agent/explorer_agent.py`), inspired by the agentic search synthesis methodology described in [DeepSeek-v3.2](https://arxiv.org/abs/2512.02556).

The pipeline generates QA pairs of varying complexity through the following stages:

1. **Concept Extraction**: Documents are sampled from the corpus and key concepts/entities are extracted using an LLM.

2. **Breadth & Depth Assignment**: Each concept is assigned `breadth` and `depth` parameters based on its occurrence frequency in the corpus:
   - **Breadth** (1-3): Controls how many related sub-questions are generated for a concept
   - **Depth** (1-3): Controls how many reasoning hops are required to answer questions

3. **Breadth Agent**: For each concept, generates multiple surface-level questions exploring different facets of the entity. The agent has access to retrieval tools to ground questions in actual document content.

4. **Depth Agent**: Takes each breadth question and iteratively deepens it, creating multi-hop questions that require chaining information across documents.

5. **Synthesis Agent**: Combines the breadth and depth QA pairs for each concept into a final, synthesized question-answer pair that captures the full complexity.

This approach ensures questions require genuine multi-hop reasoning rather than simple lookup, making them effective for training agentic search behavior.

## Task (Detailed)

### Available Tools

| Tool | Description |
|------|-------------|
| `search_tool(queries)` | Execute BM25 search queries against the document corpus and returns the most relevant chunks |
| `read_tool(doc_ids)` | Read full content of documents by their IDs |
| `text_scan_tool(pattern, ...)` | Scan documents for specific text patterns |

### Reward Functions

| Reward Function | Weight | Description |
|-----------------|--------|-------------|
| `judge_reward` | 0.8 | LLM-based judge evaluating answer correctness and completeness |
| `format_reward` | 0.2 | Compliance with expected output format |

---

## Quickstart

Run an evaluation with default settings:

```bash
prime eval run bm25-search
```

Configure model and sampling:

```bash
prime eval run bm25-search \
  -m gpt-4.1-mini \
  -n 20 \
  -r 3 \
  -t 1024 \
  -T 0.7
```

This environment does not require custom arguments, it uses the standard verifiers environment parameters.

## Metrics

| Metric | Meaning |
|--------|---------|
| `reward` | Main scalar reward (weighted sum of judge + format) |
| `judge_reward` | LLM judge score for answer quality |
| `format_reward` | Format compliance score |

---

## Active Development

This environment is under **active development**. Planned improvements include:

- Expanding the number and diversity of data sources
- Increasing the quantity of QA pairs
- Improving synthetic pipeline quality for higher quality multi-hop questions