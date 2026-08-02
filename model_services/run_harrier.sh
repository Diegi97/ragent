#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
HARRIER_PORT="${HARRIER_PORT:-3000}"

HARRIER_PID=""

cleanup() {
    trap - EXIT INT TERM
    if [[ -n "${HARRIER_PID}" ]]; then
        kill "${HARRIER_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}

trap cleanup EXIT INT TERM
cd "${SCRIPT_DIR}"

uv run bentoml serve \
    model_services.harrier_service:HarrierEmbeddingService \
    --host "${HOST}" \
    --port "${HARRIER_PORT}" &
HARRIER_PID=$!

echo "Harrier embeddings: http://${HOST}:${HARRIER_PORT}"

wait
