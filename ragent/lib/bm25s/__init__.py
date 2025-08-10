from datasets import load_dataset

from ragent.config import HUGGING_FACE_DATASET, HF_TOKEN


def populate_database():
    ds = load_dataset(HUGGING_FACE_DATASET, token=HF_TOKEN)
