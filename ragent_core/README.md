# RAGent Core

Shared data-source loaders, Turbopuffer retrieval, agent search tools, judges,
types, utilities, and operational scripts.

## Setup

Install locked runtime dependencies and configure the required API key:

```bash
uv sync --locked
cp .env-template .env
```

`TURBOPUFFER_API_KEY` is the only required retrieval setting. The defaults are
region `gcp-us-central1`, physical namespace prefix `ragent`, and logical
namespace `default`. Region and prefix can be overridden before building with
`TURBOPUFFER_REGION` and `TURBOPUFFER_NAMESPACE_PREFIX`.

Each corpus has a chunk namespace, a document namespace, and a catalog entry.
The catalog records physical resolution, schema version, exact counts, vector
availability and dimensions, embedding model, and readiness. Runtime reads
reject missing or incomplete entries. Builders never overwrite a corpus; use a
new logical namespace for a rebuild.

## Build corpora

Run scripts from the repository root. Build all Hugging Face sources:

```bash
uv run --project ragent_core python ragent_core/scripts/build_turbopuffer_indexes.py \
  --device cuda --batch-size 512
```

Build one source or a lexical-only corpus:

```bash
uv run --project ragent_core python ragent_core/scripts/build_turbopuffer_indexes.py \
  --data-source nampdn_ai_devdocs_io --device cuda --batch-size 512
uv run --project ragent_core python ragent_core/scripts/build_turbopuffer_indexes.py \
  --data-source nampdn_ai_devdocs_io --no-embedding --namespace lexical-v1
```

Build PersonaHub, optionally with `--limit 100` for a smoke test:

```bash
uv run --project ragent_core \
  python ragent_core/scripts/build_personahub_turbopuffer_index.py \
  --device cuda --batch-size 512
```

Embeddings and writes are streamed in batches. Corpus document and chunk IDs
are stored directly as Turbopuffer unsigned integers.

## Delete namespaces

Pass explicit physical namespace names. The command only previews its targets
unless `--yes` is supplied:

```bash
uv run --project ragent_core \
  python ragent_core/scripts/delete_turbopuffer_namespaces.py \
  ragent.test.example.chunks \
  ragent.test.example.documents \
  ragent.test.catalog

uv run --project ragent_core \
  python ragent_core/scripts/delete_turbopuffer_namespaces.py --yes \
  ragent.test.example.chunks \
  ragent.test.example.documents \
  ragent.test.catalog
```

## Data-source publication

Validate or upload normalized Hugging Face source data with:

```bash
uv run --project ragent_core python ragent_core/scripts/upload_data_source.py \
  posthog_com --dry-run
uv run --project ragent_core python ragent_core/scripts/upload_data_source.py posthog_com
```

Generated-data publication belongs to the
[`data_pipelines`](../data_pipelines/README.md) project.

## Package boundaries and validation

`retrievers/retriever` owns retrieval orchestration. Catalog and row contracts,
model settings, HTTP adapters, embedding/reranking backends, and index publication
live beside it under `retrievers/`. Operational scripts delegate to these package
owners. Dense and hybrid queries require the configured embedding model and
vector dimensions to agree with the catalog; missing documents return no result,
while catalog, authentication, and transport failures remain errors.

Source extensions implement `load_data_source()` in
`ragent_core.data_sources.<source>` and return a `Dataset`, a
`(Dataset, description)` tuple, or a `DataSourceSpec` (which can also set a
display name). Only a missing extension module falls back
to loading the requested Hugging Face dataset ID. Loader failures propagate. Repository
corpora use a per-source lock across preparation and loading, stage new files,
and restore the previous cache if publication fails.

Shared generated-rubric records live in `artifacts/question_rubric.py`; judge
verdicts and criterion identifiers live in `judges/criteria.py`.
`judges/evaluation` accepts current `PASS`/`FAIL` and historical `yes`/`no` teacher
traces. Run the hermetic regression suite from this directory:

```bash
uv run --locked pytest tests
```

### Import migration

Python callers should import semantic owners directly. Command-line script
paths are unchanged.

| Previous import | Current owner |
| --- | --- |
| `ragent_core.retrievers.Document`, `DocumentLike`, `RetrievalResult` | `ragent_core.retrievers.document` |
| `ragent_core.retrievers.AgentRetriever` | `ragent_core.retrievers.agent_retriever` |
| `ragent_core.retrievers.BaseRetriever` | `ragent_core.retrievers.base` |
| `ragent_core.retrievers.RetrievalMode` | `ragent_core.retrievers.mode` |
| `ragent_core.retrievers.TurbopufferRetriever` | `ragent_core.retrievers.retriever` |
| Retrieval model defaults | `ragent_core.retrievers.settings` |
| `EmbeddingServiceClient` | `ragent_core.retrievers.service_clients.embedding` |
| `CrossEncoderServiceClient` | `ragent_core.retrievers.service_clients.reranker` |
| `ModelServiceError` | `ragent_core.retrievers.model_contract` |

Replace `normalize_documents(...)` with `Document.normalize_many(...)`.
QA and entity-generation models now belong to the pipeline project:
`QA` lives in `generate.qa.models`, and `Concept` in
`data_pipelines.pipelines.deep_search_task_generation.prepare.models`.

Turbopuffer, Fireworks, and completion adapters expose `ProviderError` categories for transport, authentication,
rate-limit, API, and malformed-response failures. Turbopuffer reads and index
publication translate SDK failures at their public boundary and preserve the
original cause; a missing document still returns `None`.
