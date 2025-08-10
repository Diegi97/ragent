import os
from typing import List, Dict

import bm25s
import logging
from datasets import load_dataset
from ragent.config import HF_TOKEN
from ragent.data.pipelines import get_pipeline_run, safe_ds_name
from ragent.data.prompts.search_engine import prompt as search_engine_prompt

logger = logging.getLogger(__name__)


class BM25Client:
    _retrievers = {}
    prompt = search_engine_prompt

    def __init__(self, hf_ds):
        self.hf_ds = hf_ds
        self.load_retriever(self.hf_ds)

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
        if cls._retrievers.get(hf_ds) is None:
            index_dir = cls.ds_to_index_dir(hf_ds)
            if not os.path.exists(index_dir):
                cls._retrievers[hf_ds] = cls.populate_database(hf_ds)
            else:
                logger.info(f"Cargando retriever desde '{index_dir}'...")
                cls._retrievers[hf_ds] = bm25s.BM25.load(index_dir, load_corpus=True)
        return cls._retrievers[hf_ds]

    @staticmethod
    def format_search_results(results):
        output = []
        for i in range(results.shape[1]):
            doc = results[0, i]

            text = doc.get("text", "")
            preview_words = text.split()[:20]
            preview = " ".join(preview_words) + "..." if preview_words else ""

            output.append({
                "id": doc.get("id"),
                "title": doc.get("title", ""),
                "preview": preview
            })
        return output

    def search_tool(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, str]]:
        """
        Search the indexed corpus for the most relevant documents.

        This method:
          1. Tokenizes the query string using the same preprocessing as the corpus.
          2. Retrieves the top `k` matching documents based on BM25 scores.
          3. Transforms each retrieved document into a standardized dictionary
             containing:
             - "id": Identifier of the document in the dataset.
             - "title": Title of the document (if present; otherwise empty string).
             - "preview": The first 20 words of the document text, ending with "...".
               This provides a short snippet of content for quick reference.

        Args:
            query (str): The natural language search query.
            k (int, optional): The maximum number of results to return.
                Defaults to 5.

        Returns:
            List[Dict[str, str]]:
                A list of dictionaries, each with the following keys:
                    - "id" (Any): Document identifier.
                    - "title" (str): Document's title (or empty string if missing).
                    - "preview" (str): First 20 words of the document text, ending
                      with "...".

        Example:
            >>> results = search_tool("machine learning models", k=2)
            >>> results
            [
                {
                    "id": 42,
                    "title": "Introduction to Machine Learning",
                    "preview": "Machine learning is a field of artificial intelligence that focuses on building systems..."
                },
                {
                    "id": 7,
                    "title": "Deep Learning Architectures",
                    "preview": "Deep learning is a subfield of machine learning concerned with algorithms inspired by..."
                }
            ]
        """
        ret = self.load_retriever(self.hf_ds)
        query_tokens = bm25s.tokenize(query, stopwords="en")
        results, _ = ret.retrieve(query_tokens, k=k, corpus=ret.corpus)

        return self.format_search_results(results)

    def read_tool(self, doc_id) -> Dict[str, str] | str:
        """
        Retrieve the full document from the indexed corpus by its ID.

        Args:
            doc_id (Any): The unique identifier of the document.

        Returns:
            Dict[str, str]: A dictionary containing all available fields
                for the document (e.g., "id", "title", "text", etc.).
            Or
            str: An error message if the document is not found.

        Example:
            >>> bm25 = BM25Client("my_dataset")
            >>> bm25.read_tool(42)
            {
                "id": 42,
                "title": "Introduction to Machine Learning",
                "text": "Machine learning is a field of artificial intelligence that..."
            }
            >>> bm25.read_tool(9999)
            "Error: Document with id '9999' not found in corpus."
        """
        ret = self.load_retriever(self.hf_ds)

        for doc in ret.corpus:
            if doc.get("id") == doc_id:
                return doc

        return f"Error: Document with id '{doc_id}' not found in corpus."

if __name__ == "__main__":
    res = BM25Client.search_tool("What is the way for using a Django Serializer?")
    for doc in res:
        logger.info(f"Found doc: {doc}\n\n")
