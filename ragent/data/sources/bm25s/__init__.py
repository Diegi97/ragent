import os
from typing import List, Tuple

import bm25s
import logging
from datasets import load_dataset
from ragent.config import HF_TOKEN
from ragent.data.pipelines import get_pipeline_run, safe_ds_name

logger = logging.getLogger(__name__)


class BM25Client:
    _retriever = {}

    @staticmethod
    def ds_to_index_dir(ds_name):
        return f"data/{safe_ds_name(ds_name)}/bm25s_corpus_index/"

    @classmethod
    def populate_database(cls, hf_ds, split="train", corpus_column="text", force=False):
        index_dir = cls.ds_to_index_dir(hf_ds)

        if not force and os.path.exists(index_dir):
            logger.info(f"Index: '{index_dir}' already exists, skipping.")
            return

        dataset = load_dataset(hf_ds, token=HF_TOKEN)
        pipeline_run = get_pipeline_run(hf_ds)

        dataset_split_clean = pipeline_run(dataset[split])

        corpus = dataset_split_clean[corpus_column]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        retriever.save(index_dir, corpus=corpus)
        logger.info("Index successfully created.")
        return retriever

    @classmethod
    def load_retriever(cls, hf_ds):
        """Carga el retriever en memoria si no está cargado todavía."""
        if cls._retriever.get(hf_ds) is None:
            index_dir = cls.ds_to_index_dir(hf_ds)
            if not os.path.exists(index_dir):
                cls._retriever = cls.populate_database(hf_ds)
            else:
                logger.info(f"Cargando retriever desde '{index_dir}'...")
                cls._retriever = bm25s.BM25.load(index_dir, load_corpus=True)
        return cls._retriever

    @classmethod
    def search_tool(
        cls,
        query: str,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Search the indexed corpus for the most relevant documents.

        This method:
          1. Tokenizes the query string using the same preprocessing as the corpus.
          2. Retrieves the top `k` matching documents based on BM25 scores.
          3. Returns each match as a tuple containing:
             - A dictionary with:
               - "id" (Any): Identifier of the document in the dataset.
               - "text" (str): Full text content of the document.
             - The BM25 relevance score as a float.

        Args:
            query (str): The natural language search query.
            k (int, optional): The maximum number of results to return.
                Defaults to 5.

        Returns:
            List[Tuple[Dict[str, Any], float]]:
                A list of `(document_metadata, score)` tuples.
                `document_metadata` has keys:
                    - "id" (Any): Document identifier.
                    - "text" (str): Document's text content.
                The `score` is the BM25 relevance score (higher means more relevant).

        Example:
            >>> results = search_tool("machine learning models", k=2)
            >>> results
            [
                ({"id": 42, "text": "Intro to Machine Learning..."}, 5.43),
                ({"id": 7, "text": "Deep learning architectures..."}, 4.98)
            ]
        """

        retriever = cls.load_retriever()

        query_tokens = bm25s.tokenize(query, stopwords="en")
        results, scores = retriever.retrieve(query_tokens, k=k, corpus=retriever.corpus)

        return [
            (results[0, i], scores[0, i])
            for i in range(results.shape[1])
        ]


if __name__ == "__main__":
    resultados = BM25Client.search("What is the way for using a Django Serializer?", k=3)
    for rank, (doc, score) in enumerate(resultados, start=1):
        logger.info(f"Rank {rank} (score: {score:.2f}): {doc}")
