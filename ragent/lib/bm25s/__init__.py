import os
import bm25s
import logging
from datasets import load_dataset

from ragent.config import HUGGING_FACE_DATASET, HF_TOKEN

logger = logging.getLogger(__name__)

INDEX_DIR = f"data/bm25s_index/{HUGGING_FACE_DATASET}"


def populate_database(force=False):
    """
    Create a BM25 index from the Hugging Face dataset.

    Parameters:
        force (bool): If False and index already exists, skip creation.
    """
    if not force and os.path.exists(INDEX_DIR):
        logger.info(f"Index already exists at '{INDEX_DIR}'. Skipping creation.")
        return

    logger.info("Loading dataset...")
    dataset = load_dataset(HUGGING_FACE_DATASET, token=HF_TOKEN)
    corpus = dataset["train"]["text"]

    logger.info("Tokenizing corpus (no stemmer)...")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

    logger.info("Creating BM25 retriever...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    logger.info(f"Saving index to '{INDEX_DIR}'...")
    retriever.save(INDEX_DIR, corpus=corpus)
    logger.info("Index creation complete.")


def get_retriever():
    """
    Load an existing BM25 retriever with corpus.
    """
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(
            f"Index directory '{INDEX_DIR}' not found. Run populate_database(force=True) first."
        )

    logger.info(f"Loading retriever from '{INDEX_DIR}'...")
    retriever = bm25s.BM25.load(INDEX_DIR, load_corpus=True)
    return retriever