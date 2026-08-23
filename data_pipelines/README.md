# Data Pipelines

This project generates synthetic data for evaluating and training agentic search systems. The deep-search task-generation pipeline is the primary pipeline; retrieval-query generation and retrieval evaluation are supporting experiments.

## Pipelines

### 1. Deep-search task generation

Creates grounded, multi-document questions and rubrics for agents that must search a defined knowledge base, connect evidence across documents, and synthesize an answer.

The pipeline is designed around one **main goal: answer uniqueness**. An incorrect golden answer or rubric becomes an evaluation and training bug, so the pipeline brute-forces grounding by gathering all discoverable documents and facts about an entity. It then trusts a competent agent to construct a question and rubric whose answer is uniquely supported by that evidence.

General generation logic:

1. Sample entities from a selected data source.
2. Retrieve **all relevant documents** for each entity from the data source.
3. Extract **all facts** about each entity from those documents, retaining the supporting document IDs.
4. Link entity mentions into a soft knowledge graph and give the assigned entity plus the wider extracted graph to a [Pi](https://github.com/earendil-works/pi) synthesizer agent.
5. Have the synthesizer draft a natural base question and an explicit, document-grounded rubric, then validate its format, evidence IDs, and answer uniqueness.
6. Run a retrieval probe with the question itself. If every supporting document appears in the top 10, the question is too lexically transparent, so the synthesizer evolves it before spending a solver rollout.
7. Once the retrieval gate passes, run one solver rollout for that candidate version. A rubric judge returns the solver's answer, per-criterion judgments, and the percentage of criteria satisfied.
8. Use the retrieval and solver evidence to calibrate difficulty. When an item is too easy, apply one evolution strategy, then revalidate, reprobe, and audit uniqueness before solving the revised version.
9. Stop at the entity's attainable difficulty frontier, or when an integrity, fact, or step budget fires. Reject or repair ambiguous and near-zero-score items rather than treating broken tasks as hard, then aggregate valid records into the final dataset.

The synthesizer-solver loop and its difficulty-evolution strategies were inspired by Prime Intellect's [General Agent](https://www.primeintellect.ai/blog/general-agent).

The same prepared entity facts can alternatively be used to generate conventional question-answer records.

### 2. Retrieval-query generation

Creates queries that resemble what a search **agent** would issue while investigating a task. They are intentionally not modeled after queries written by human search-engine users.

General generation logic:

1. Sample a source chunk from the selected data source.
2. Generate a query for one information need answered by that chunk, without revealing the answer.
3. Retrieve competing chunks from the same corpus.
4. Add the smallest disambiguating detail needed to separate the positive chunk from those candidates.
5. Save the target as the positive example and suitable competing results as hard negatives.

The complete generation and contrastive-narrowing contracts are defined in [`prompts.py`](data_pipelines/pipelines/search_query_generation/prompts.py).

### 3. Retrieval evaluation

Evaluates a completed retrieval-query run against one dense, BM25, hybrid, or reranked Turbopuffer configuration. It reports chunk- and document-level ranking quality, coverage, and latency.

## Architecture and technologies

- **Prefect** orchestrates flows, tracks execution state, and limits concurrent LLM and retriever calls.
- **Phoenix** provide object-level traces and instrument OpenAI requests and responses.
- **JSONL** is the canonical generated dataset output file format.
- **Turbopuffer and the RAGent retriever** provide source sampling, retrieval, and hard-negative mining.
- **Fireworks batch inference** extracts entity facts for deep-search task generation.
- **Agents** generate and validate question-rubric records from grounded entity facts using the PI harness.

Generated objects run independently. Each object receives a stable ID, its own Prefect child flow and Phoenix trace, and writes its result immediately under a cross-process file lock. Child flows run concurrently, while global Prefect limits bound external calls. A failure in one object does not stop the rest of the batch.

Pipeline code runs in the local `uv` environment; Prefect and Phoenix run as local Docker services. No long-running Prefect deployment is required. Phoenix tracing is best-effort and does not block dataset generation when unavailable.

Deep-search preparation uses a standalone sequential retriever with models loaded in-process. Concurrent requests enter a FIFO buffer and are processed as complete, predictable work units, making it straightforward to keep the GPU saturated without complex web-server concurrency that can overload the model or produce uneven work. Retrieval-query generation instead uses remote embedding services so concurrent flows can share one deployment.

## Local setup

```bash
uv sync
cp .env-template .env
docker compose up -d
```

The local interfaces are available at:

- Prefect: <http://127.0.0.1:4200>
- Phoenix: <http://127.0.0.1:6006>

All pipelines require ready Turbopuffer catalog entries. Set `TURBOPUFFER_API_KEY`; region, physical prefix, and logical namespace default to `gcp-us-central1`, `ragent`, and `default`. Configure provider credentials and model-service URLs in the uncommitted `.env` file.

## Run deep-search task generation

Deep-search generation is split around an external Fireworks batch job.

### 1. Start the retriever worker

Set a local authentication key and start the sequential retriever in one terminal:

```bash
export RAGENT_RETRIEVER_AUTHKEY='replace-with-a-local-random-secret'
uv run deep-search-tasks retriever \
  --retriever-namespace default \
  --retriever-device cuda \
  --rerank-threshold 3.0
```

The worker binds to localhost, loads one `TurbopufferRetriever`, buffers incoming requests in a FIFO queue, and processes one complete retrieval request at a time. This keeps work units predictable and lets each request use the GPU efficiently without exposing the model to uncontrolled concurrent load. `--num-chunks-per-entity` is the maximum candidate count presented to the CrossEncoder; only candidates passing `--rerank-threshold` are returned.

### 2. Prepare entities and fact requests

In another terminal, use the same authentication key:

```bash
export RAGENT_RETRIEVER_AUTHKEY='replace-with-a-local-random-secret'
uv run deep-search-tasks prepare \
  --data-source nampdn_ai_devdocs_io \
  --num-entities 10
```

The command creates a prepare-run directory and a Fireworks input dataset named `deep-search-tasks-<data-source>-<UTC-timestamp>-<retained-entities>e`. Launch the fact-extraction batch in Fireworks and wait for its output dataset; the pipeline deliberately does not create or poll that external job.

To continue from entities produced by an earlier prepare run, pass its entity file:

```bash
uv run deep-search-tasks prepare \
  --data-source nampdn_ai_devdocs_io \
  --entities-file data/deep_search_task_generation/<prepare-run>/entities.jsonl \
  --num-entities 10
```

This skips entity extraction and loads up to `--num-entities` unique records in file order. Each reused entity must have the requested data source and reference a document ID that still exists in the corpus.

For a smoke test, prepare one entity with `--num-entities 1` and confirm that `prepare_metadata.json` contains the retriever configuration.

### 3. Generate question-rubric records

Finalize the Fireworks output with Pi agents:

```bash
uv run deep-search-tasks generate-rubrics \
  --prepare-dir data/deep_search_task_generation/<prepare-run> \
  --batch-dataset <fireworks-output-dataset> \
  --model accounts/fireworks/models/kimi-k3 \
  --solver-model deepseek/deepseek-v4-flash-0731 \
  --num-rubrics 100 \
  --pi-concurrency 10 \
  --max-attempts 4
```

Pi and the `pi-phoenix` package must be installed and configured in the environment running the flow (`pi install npm:pi-phoenix`). Each agent writes a Markdown question-rubric record.

The requested rubric count is distributed round-robin over the prepared entities. Fireworks operations require `OPENAI_API_KEY` and `FIREWORKS_ACCOUNT_ID`. The retriever defaults to port `8765`; when overriding it, pass matching ports to `retriever` and `prepare`.

### Alternative: generate QA records

```bash
uv run deep-search-tasks generate-qas \
  --prepare-dir data/deep_search_task_generation/<prepare-run> \
  --batch-dataset <fireworks-output-dataset> \
  --qa-model-id accounts/fireworks/models/kimi-k3 \
  --qa-pairs-per-entity 4 \
  --llm-concurrency 25
```

### Deep-search outputs

Prepare runs are written under `data/deep_search_task_generation/` and contain:

```text
<prepare-run>/
├── entities.jsonl
├── fact_requests.jsonl
├── failures.jsonl
└── prepare_metadata.json
```

Each rubric finalization creates an immutable `rubric_finalize_*` child directory with the raw Fireworks files, extracted `entity_facts.jsonl`, final `question_rubrics.jsonl`, failures, and metadata. QA generations create a separate child directory with equivalent provenance and `qas.jsonl`.

Upload question-rubric records to Hugging Face with train and test splits:

```bash
uv run python scripts/upload_question_rubrics.py \
  data/deep_search_task_generation/<prepare-run>/<rubric-run>/question_rubrics.jsonl \
  <data-source>
```

The default destination is the private `diegi97/ragent-rubrics` dataset with a 10% test split. Use `--replace-data`, `--test-size`, or a third positional dataset ID to override those defaults.

## Run retrieval-query generation

`RAGENT_EMBEDDING_SERVICE_URL` must point to a running embedding service. The pipeline also requires `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`.

```bash
uv run generate-search-queries run \
  --table-name nampdn_ai_devdocs_io \
  --num-queries 10
```

Useful overrides include `--generator-model`, `--logical-namespace`, `--llm-concurrency`, `--retriever-concurrency`, and `--output-path`. Run with `--num-queries 1`, `--candidate-mining-top-k 5`, and `--round-trip-top-k 5` for a small smoke test.

Every run creates a unique directory under `data/search_query_generation/`:

```text
<table-name>_<timestamp>_<query-count>q_<prefect-run-id>/
├── queries.jsonl
├── failures.jsonl
└── metadata.json
```

Records contain the complete positive and hard-negative document text. Phoenix traces retain the generation stages, OpenAI requests and responses, model information, and token usage.

## Run retrieval evaluation

Retrieval evaluation is synchronous and does not require Prefect or Phoenix. It reads a completed generation directory and requires the table and logical namespace recorded in its metadata, preventing accidental evaluation against another corpus.

```bash
uv run evaluate-retrieval run \
  --input-directory data/search_query_generation/<generation-run> \
  --search-type hybrid \
  --top-k 50 \
  --cutoffs 1,3,5,10,20
```

Dense and hybrid evaluation use the local embedding model by default or the service configured by `--embedding-service-url` or `RAGENT_EMBEDDING_SERVICE_URL`. Add `--reranker` to hybrid retrieval and optionally select a remote reranker with `--reranker-service-url` or `RAGENT_RERANKER_SERVICE_URL`.

Each run creates a child directory containing `summary.json` and `details.jsonl`. The summary reports configuration, provenance, coverage, latency, Precision, Recall/Hit Rate, MRR, MAP, and nDCG at each cutoff. Metrics are calculated for both exact positive chunks and documents after collapsing duplicate chunks.

Evaluation continues after individual retrieval errors and writes partial artifacts, but exits with status 1 if any query failed. Invalid input or configuration exits with status 2 before evaluation.

## Back up generated data to GCS

Set the destination in `.env`:

```dotenv
DATA_PIPELINES_GCS_URI=gs://your-bucket/ragent/data-pipelines
# Optional when existing gcloud credentials are not used:
GCS_SERVICE_ACCOUNT=/absolute/path/to/service-account.json
```

Preview and synchronize the complete `data/` directory:

```bash
uv run python scripts/sync_data_gcs.py upload --dry-run
uv run python scripts/sync_data_gcs.py upload
```

Use `download` to restore data and `--keep-extra` to retain destination-only files. By default the script uses `gcloud storage rsync --delete-unmatched-destination-objects`, making the destination an exact mirror. The Google Cloud CLI must be installed.

## Troubleshooting cancelled runs

Prefect concurrency slots can remain occupied if a run is cancelled and then interrupted before shutdown cleanup finishes. A later run may appear stuck before an LLM operation.

Inspect the relevant limit:

```bash
uv run python -m prefect gcl ls
uv run python -m prefect gcl inspect deep-search-tasks-openai-llm
```

Retrieval-query generation uses the `local-retriever` and `openai-llm` limits. Do not reset an occupied limit while its original run is still active. When cancelling normally, press `Ctrl+C` once and let Prefect finish releasing its leases.
