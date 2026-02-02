"""
Merge multiple QA dataset directories into a single combined dataset.

This script takes multiple QA dataset directories (each containing data.json and metadata.json),
merges them into a single dataset, and adds a dataset_id field to each QA pair.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def load_dataset(dataset_path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load metadata and data from a dataset directory."""
    metadata_file = dataset_path / "metadata.json"
    data_file = dataset_path / "data.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    with open(data_file, "r") as f:
        data = json.load(f)

    return metadata, data


def merge_datasets(dataset_paths: List[str], output_dir: str) -> None:
    """
    Merge multiple QA datasets into a single combined dataset.

    Args:
        dataset_paths: List of paths to dataset directories
        output_dir: Directory where merged dataset will be saved
    """
    all_qa_pairs = []
    source_datasets = []
    total_pairs = 0
    id_counter = 1

    print(f"Merging {len(dataset_paths)} datasets...")

    for dataset_path_str in dataset_paths:
        dataset_path = Path(dataset_path_str)

        if not dataset_path.exists():
            print(f"Warning: Dataset path does not exist: {dataset_path}")
            continue

        print(f"\nProcessing: {dataset_path}")

        metadata, data = load_dataset(dataset_path)

        # Extract dataset identifier from metadata
        dataset_id = metadata.get(
            "dataset_id", metadata.get("run_identifier", dataset_path.name)
        )
        run_identifier = metadata.get("run_identifier", dataset_path.name)

        print(f"  - Dataset ID: {dataset_id}")
        print(f"  - Run: {run_identifier}")
        print(f"  - QA pairs: {len(data)}")

        # Add dataset_id to each QA pair and renumber
        for qa_pair in data:
            qa_pair["dataset_id"] = dataset_id
            qa_pair["source_run"] = run_identifier
            qa_pair["original_id"] = qa_pair["id"]
            qa_pair["id"] = id_counter
            id_counter += 1
            all_qa_pairs.append(qa_pair)

        # Track source dataset info
        source_datasets.append(
            {
                "dataset_id": dataset_id,
                "run_identifier": run_identifier,
                "path": str(dataset_path),
                "created_at": metadata.get("created_at"),
                "num_pairs": len(data),
                "models": metadata.get("models"),
                "pipeline": metadata.get("pipeline"),
                "retriever_type": metadata.get("retriever_type"),
                "parameters": metadata.get("parameters"),
            }
        )

        total_pairs += len(data)

    if not all_qa_pairs:
        print("\nNo QA pairs to merge. Exiting.")
        return

    # Update output directory name to include total pairs count
    output_base = Path(output_dir)
    output_path = Path(f"{output_base}-{total_pairs}pairs")
    output_path.mkdir(parents=True, exist_ok=True)

    # Create merged metadata
    merged_metadata = {
        "created_at": datetime.now().isoformat(),
        "merge_type": "multi_dataset",
        "total_pairs": total_pairs,
        "num_source_datasets": len(source_datasets),
        "source_datasets": source_datasets,
        "output": {
            "data_path": str(output_path / "data.json"),
            "metadata_path": str(output_path / "metadata.json"),
        },
    }

    # Write merged data
    data_output = output_path / "data.json"
    with open(data_output, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)

    # Write merged metadata
    metadata_output = output_path / "metadata.json"
    with open(metadata_output, "w") as f:
        json.dump(merged_metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Merge complete!")
    print(f"{'=' * 60}")
    print(f"Total QA pairs: {total_pairs}")
    print(f"Source datasets: {len(source_datasets)}")
    print("\nOutput files:")
    print(f"  - Data: {data_output}")
    print(f"  - Metadata: {metadata_output}")


if __name__ == "__main__":
    # Hardcoded list of dataset paths to merge
    # Update these paths as needed
    DATASET_PATHS = [
        "data/gitlab_handbook/qa-multiagent-20260117-201550",
        "data/gitlab_handbook/qa-multiagent-20260118-132612",
        "data/gitlab_handbook/qa-multiagent-20260118-220936",
        "data/rfc_all/qa-multiagent-20260120-163735",
        "data/rfc_all/qa-multiagent-20260120-202228",
        "data/common_pile_peps/qa-multiagent-20260123-214305",
        "data/common_pile_peps/qa-multiagent-20260123-215948",
        "data/common_pile_peps/qa-multiagent-20260124-000414",
        "data/nampdn_ai_devdocs_io/qa-multiagent-20260124-214209",
        "data/nampdn_ai_devdocs_io/qa-multiagent-20260125-123928/",
        "data/diegi97_marine_biology/qa-multiagent-20260125-041909",
        "data/diegi97_space_exploration/qa-multiagent-20260125-134632",
        "data/owasp_cheatsheets/qa-multiagent-20260125-170631",
        "data/rust_rfcs/qa-multiagent-20260125-233930",
        "data/diegi97_mythology/qa-multiagent-20260126-092651",
        "data/owasp_cheatsheets/qa-multiagent-20260127-232546",
        "data/rfc_all/qa-multiagent-20260128-074812",
        "data/rust_rfcs/qa-multiagent-20260127-221250",
        "data/nampdn_ai_devdocs_io/qa-multiagent-20260129-140852",
    ]

    # Output directory for merged dataset
    OUTPUT_DIR = "data/merged/qa-merged-" + datetime.now().strftime("%Y%m%d-%H%M%S")

    print("QA Dataset Merger")
    print("=" * 60)
    print("Datasets to merge:")
    for path in DATASET_PATHS:
        print(f"  - {path}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 60)

    merge_datasets(DATASET_PATHS, OUTPUT_DIR)
