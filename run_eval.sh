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
uv run vf-eval ragent.environments.${ENV} --endpoints-path ragent/config/endpoints.py --model $MODEL --max-tokens 32768 --save-dataset --save-path "results/${MODEL}_${ENV}/"

# This saves the dataset in a format that can be used by the visualizer index.html
cat <<EOF | env MODEL="${MODEL}" ENV="${ENV}" uv run --with datasets -
import json
from datasets import load_from_disk
import os

path = f"results/{os.environ['MODEL']}_{os.environ['ENV']}/"
ds = load_from_disk(path)
ds_dict = ds.to_dict()

with open(f"{path}trajectories.json", "w") as f:
    json.dump(ds_dict, f, indent=4)
EOF
