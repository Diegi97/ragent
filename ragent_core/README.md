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
