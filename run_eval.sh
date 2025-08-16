#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Run the vf-eval command
uv run vf-eval ragent.environments.bm25s --endpoints-path ragent/utils/config.py --model qwen3-a22b --max-tokens 32768 --save-dataset
