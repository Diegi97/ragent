# Model Services

BentoML services for the Harrier embedding model and mxbai CrossEncoder used by
the Turbopuffer retriever. BentoML provides flexibility over the served models
without tying the project to Infinity or TEI.

Use these services for request-parallel workloads, especially search-query
generation and concurrent environment rollouts. They let many retrieval calls
share and independently scale one model deployment. For deep-search task
preparation and index-building scripts, prefer loading models in the Python process: those
workloads are sequential or batch-oriented, so direct access avoids HTTP
overhead and keeps batching under the caller's control.

## Run locally

```bash
uv sync
./run_harrier.sh   # Harrier embeddings on port 3000
./run_mxbai.sh     # mxbai reranker on port 3001
```

The Harrier service listens on port `3000` and the mxbai service on `3001`.
Override them with `HARRIER_PORT`, `MXBAI_PORT`, and `HOST`.

```bash
curl http://127.0.0.1:3000/v1/embeddings/query \
  -H 'content-type: application/json' \
  -d '{"texts":["how much protein should a female eat"]}'

curl http://127.0.0.1:3000/v1/embeddings/documents \
  -H 'content-type: application/json' \
  -d '{"texts":["The recommended daily protein intake is 46 grams."]}'

curl http://127.0.0.1:3001/v1/rerank \
  -H 'content-type: application/json' \
  -d '{"query":"daily protein intake","texts":["46 grams per day","A weather report"]}'
```

Query embeddings use Harrier's `web_search_query` prompt. Document embeddings
are deliberately encoded without a prompt. Both outputs are L2-normalized.
The inference libraries are pinned in this project's `uv.lock` so existing
index embeddings and raw reranker thresholds stay reproducible.

## Configuration

| Variable | Default |
| --- | --- |
| `HARRIER_MODEL_ID` | `microsoft/harrier-oss-v1-0.6b` |
| `HARRIER_DEVICE` | `auto` |
| `HARRIER_MAX_SEQ_LENGTH` | `512` |
| `HARRIER_INFERENCE_BATCH_SIZE` | `8` |
| `HARRIER_MAX_BATCH_SIZE` | `16` |
| `HARRIER_MAX_LATENCY_MS` | `25` |
| `HARRIER_TIMEOUT_SECONDS` | `60` |
| `MXBAI_MODEL_ID` | `mixedbread-ai/mxbai-rerank-base-v2` |
| `MXBAI_DEVICE` | `auto` |
| `MXBAI_MAX_LENGTH` | `512` |
| `MXBAI_INFERENCE_BATCH_SIZE` | `8` |
| `MXBAI_TIMEOUT_SECONDS` | `60` |

Set `RAGENT_EMBEDDING_SERVICE_URL=http://127.0.0.1:3000` and optionally
`RAGENT_RERANKER_SERVICE_URL=http://127.0.0.1:3001` in the process running
ragent. The clients read timeout and retry settings from environment
variables, with these defaults: `RAGENT_MODEL_SERVICE_TIMEOUT=60`,
`RAGENT_MODEL_SERVICE_CONNECT_TIMEOUT=5`, and
`RAGENT_MODEL_SERVICE_MAX_RETRIES=2`. Requests use two retries (three total
attempts) for
timeouts, network errors, and HTTP 408/429/502/503/504 responses, with
exponential full-jitter backoff (1 second base, capped at 30 seconds) and
support for `Retry-After`. Override the backoff with
`RAGENT_MODEL_SERVICE_RETRY_BASE_SECONDS` or
`RAGENT_MODEL_SERVICE_RETRY_MAX_SECONDS`. Permanent 4xx responses and
malformed successful responses are not retried. The retrieval-query pipeline
always uses RRF-only retrieval and does not load the reranker.
