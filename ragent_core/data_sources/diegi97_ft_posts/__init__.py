import logging

from datasets import Dataset


logger = logging.getLogger(__name__)


def run(dataset: Dataset):
    logger.info(f"Running pipeline over {dataset}")
    dataset = dataset.map(lambda x: {"text": f"# {x['title']}\n\n" + x["content"]})
    return dataset
