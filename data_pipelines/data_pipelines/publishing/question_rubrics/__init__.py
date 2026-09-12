from datasets import DatasetDict, load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import HfApi

from ragent_core.config import HF_TOKEN


def load_remote_dataset(dataset_id: str) -> DatasetDict | None:
    """Load an existing Hub dataset, returning None when it does not exist."""
    try:
        dataset = load_dataset(dataset_id, token=HF_TOKEN)
    except DatasetNotFoundError:
        return None
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected {dataset_id} to load as a DatasetDict, got {type(dataset).__name__}"
        )
    return dataset


def publish_dataset(dataset: DatasetDict, dataset_id: str, data_source: str) -> None:
    """Create or update a Hub dataset and enforce private visibility."""
    api = HfApi(token=HF_TOKEN)
    api.create_repo(
        dataset_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    api.update_repo_settings(
        dataset_id,
        repo_type="dataset",
        private=True,
    )
    dataset.push_to_hub(
        dataset_id,
        private=True,
        token=HF_TOKEN,
        commit_message=f"Update question rubrics: {data_source}",
    )
