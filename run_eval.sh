#!/bin/bash

# Example usage: ./run_eval.sh gpt-4.1-mini bm25s
# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Run the vf-eval command
MODEL=${1:-gpt-4.1-mini}
ENV=${2:-bm25s} # Extracting environment from "ragent.environments.bm25s"
echo "Running evaluation for $MODEL"
uv run vf-eval ragent.environments.${ENV} --endpoints-path ragent/utils/config.py --model $MODEL --max-tokens 32768 --save-dataset --save-path "results/${MODEL}_${ENV}/"
