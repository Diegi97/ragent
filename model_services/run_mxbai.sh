#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
MXBAI_PORT="${MXBAI_PORT:-3001}"

MXBAI_PID=""

cleanup() {
    trap - EXIT INT TERM
    if [[ -n "${MXBAI_PID}" ]]; then
        kill "${MXBAI_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}

trap cleanup EXIT INT TERM
cd "${SCRIPT_DIR}"

uv run bentoml serve \
    model_services.mxbai_reranker_service:MxbaiRerankerService \
    --host "${HOST}" \
    --port "${MXBAI_PORT}" &
MXBAI_PID=$!

echo "mxbai reranker: http://${HOST}:${MXBAI_PORT}"

wait
