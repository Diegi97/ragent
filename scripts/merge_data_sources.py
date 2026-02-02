"""
Script to merge all data sources into a single HuggingFace dataset with each
data source as a separate split, then upload to HuggingFace hub.

Usage:
    uv run python scripts/merge_data_sources.py
"""

import logging
from datasets import DatasetDict
from ragent_core.data_sources import (
    get_data_source_loader,
    normalize_data_source_result,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Data source module names (will be used as split names)
DATA_SOURCES = [
    "common_pile_peps",
    "diegi97_marine_biology",
    "diegi97_mythology",
    "diegi97_space_exploration",
    "gitlab_handbook",
    "nampdn_ai_devdocs_io",
    "owasp_cheatsheets",
    "rfc_all",
    "rust_rfcs",
]

HF_REPO_ID = "diegi97/ragent_data_sources"


def load_all_data_sources() -> DatasetDict:
    """
    Load all data sources and return them as a DatasetDict where each
    data source becomes a split.

    Returns:
        DatasetDict with each data source as a separate split
    """
    splits = {}

    for module_name in DATA_SOURCES:
        logger.info(f"Loading data source: {module_name}")
        loader = get_data_source_loader(module_name)
        result = loader()
        spec = normalize_data_source_result(result)

        # Store the dataset in the splits dict with module_name as split name
        splits[module_name] = spec.dataset

        logger.info(f"  ✓ Loaded {module_name}: {len(spec.dataset)} documents")
        if spec.description:
            logger.info(f"  Description: {spec.description}")

    return DatasetDict(splits)


def upload_to_huggingface(dataset_dict: DatasetDict, repo_id: str):
    """
    Upload the merged dataset to HuggingFace Hub.

    Args:
        dataset_dict: DatasetDict containing all data source splits
        repo_id: HuggingFace repository ID (e.g., "diegi97/ragent_data_sources")
    """

    logger.info(f"Uploading dataset to HuggingFace: {repo_id}")

    # Create dataset card description
    card_description = """
# Data Sources Dataset

This dataset contains ground truth documents from multiple data sources used for 
agentic search RL training. Each split represents 
a different data source.

## Splits

"""

    for split_name, dataset in dataset_dict.items():
        card_description += f"- **{split_name}**: {len(dataset)} documents\n"

    card_description += """
## Schema

Each document has the following fields:
- `id`: Unique identifier within the split
- `title`: Document title (may be None)
- `text`: Full document text with markdown formatting

## Sources

- **common_pile_peps**: Python Enhancement Proposals
- **diegi97_marine_biology**: Wikipedia articles on marine biology
- **diegi97_mythology**: Wikipedia articles on mythology
- **diegi97_space_exploration**: Wikipedia articles on space exploration
- **gitlab_handbook**: GitLab company handbook
- **nampdn_ai_devdocs_io**: Technical documentation from devdocs.io
- **owasp_cheatsheets**: OWASP security cheat sheets
- **rfc_all**: IETF Request for Comments (RFCs)
- **rust_rfcs**: Rust RFCs

## Usage

```python
from datasets import load_dataset

# Load all splits
dataset = load_dataset("diegi97/ragent_data_sources")

# Load a specific split
peps = load_dataset("diegi97/ragent_data_sources", split="common_pile_peps")
```
"""

    # Push to hub with dataset card
    dataset_dict.push_to_hub(
        repo_id,
        private=True,
        commit_message="Upload merged data sources dataset",
    )

    logger.info(f"✓ Successfully uploaded to {repo_id}")

    # Print summary
    logger.info("\nDataset Summary:")
    logger.info(f"Repository: https://huggingface.co/datasets/{repo_id}")
    logger.info(f"Total splits: {len(dataset_dict)}")
    total_docs = sum(len(ds) for ds in dataset_dict.values())
    logger.info(f"Total documents: {total_docs}")
    logger.info("\nSplit details:")
    for split_name, dataset in dataset_dict.items():
        logger.info(f"  {split_name}: {len(dataset)} documents")


def main():
    """Main entry point for the script."""
    logger.info("=" * 80)
    logger.info("Merging Data Sources Dataset")
    logger.info("=" * 80)

    # Load all data sources
    logger.info("\nStep 1: Loading all data sources...")
    dataset_dict = load_all_data_sources()

    # Upload to HuggingFace
    logger.info("\nStep 2: Uploading to HuggingFace...")
    upload_to_huggingface(dataset_dict, HF_REPO_ID)

    logger.info("\n" + "=" * 80)
    logger.info("✓ Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
