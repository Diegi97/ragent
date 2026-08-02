# RAGent Core

Shared components for the RAGent pipelines and environments, including data-source loaders, LanceDB retrieval, agent search tools, judges, common types, and operational scripts.

## Setup

From this directory, install the locked dependencies:

```bash
uv sync
```

Copy `.env-template` to an uncommitted `.env` when credentials or remote storage configuration are needed.

## Operational scripts

Run the scripts from the **repository root** with `uv run --project ragent_core`. The default LanceDB URI is relative, so this keeps local indexes in the repository-level `lancedb/` directory rather than under `ragent_core/`.

### Build LanceDB indexes

Build chunk and document tables for every split in `diegi97/ragent_data_sources`:

```bash
uv run --project ragent_core python ragent_core/scripts/build_lancedb_indexes.py \
  --device cuda --batch-size 512
```

Build one source only:

```bash
uv run --project ragent_core python ragent_core/scripts/build_lancedb_indexes.py \
  --data-source nampdn_ai_devdocs_io --device cuda --batch-size 512
```

Create lexical BM25/FTS and scalar indexes without embeddings:

```bash
uv run --project ragent_core python ragent_core/scripts/build_lancedb_indexes.py \
  --data-source nampdn_ai_devdocs_io --no-embedding
```

Each source produces `<data_source>_chunks` and `<data_source>_documents` tables under `lancedb/<namespace>/`. Use `--namespace` to select another namespace.

### Build the PersonaHub diversity index

PersonaHub is intended as a future diversity pool for synthetic-data generation, not as a searchable source corpus. Its pipeline integration is still a work in progress and it is not currently used.

```bash
uv run --project ragent_core python ragent_core/scripts/build_personahub_lancedb_index.py \
  --device cuda --batch-size 512
```

Use `--limit 100` for a small smoke test.

### Sync indexes with GCS

GCS can act as the durable store while local LanceDB remains the hot read path. Configure `LANCEDB_GCS_URI` and, if needed, `GCS_SERVICE_ACCOUNT`, then upload or download a namespace:

```bash
uv run --project ragent_core python ragent_core/scripts/sync_lancedb_gcs.py upload \
  --namespace default
uv run --project ragent_core python ragent_core/scripts/sync_lancedb_gcs.py download \
  --namespace default
```

Synchronization mirrors the source namespace to the destination. The script requires `gsutil`, plus `gcloud` when activating a service account.

### Upload a data source

Data-source modules normalize corpora to the shared `id`, `title`, and `text` schema. Validate one locally:

```bash
uv run --project ragent_core python ragent_core/scripts/upload_data_source.py \
  posthog_com --dry-run
```

Upload it as an independent Hugging Face dataset split:

```bash
uv run --project ragent_core python ragent_core/scripts/upload_data_source.py posthog_com
```

Pass `--overwrite` to replace that source's split while preserving the others, or `--repo-id owner/dataset` to target another dataset.

Question-rubric publication and generated-data synchronization belong to the [`data_pipelines`](../data_pipelines/README.md) project.
