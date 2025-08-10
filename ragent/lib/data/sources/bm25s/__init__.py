import os
import bm25s
import logging
from datasets import load_dataset
from ragent.config import HUGGING_FACE_DATASET, HF_TOKEN
from ragent.lib.data.pipelines import get_pipeline_run

logger = logging.getLogger(__name__)

INDEX_DIR = f"data/bm25s_index/{HUGGING_FACE_DATASET}"


class BM25Client:
    _retriever = None  # atributo de clase

    @classmethod
    def populate_database(cls, split="train", corpus_column="text", force=False):
        if not force and os.path.exists(INDEX_DIR):
            logger.info(f"Index: '{INDEX_DIR}' already exists, skipping.")
            return

        dataset = load_dataset(HUGGING_FACE_DATASET, token=HF_TOKEN)
        pipeline_run = get_pipeline_run(HUGGING_FACE_DATASET)

        dataset_split_clean = pipeline_run(dataset[split])

        corpus = dataset_split_clean[corpus_column]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        retriever.save(INDEX_DIR, corpus=corpus)
        logger.info("Index succesfully created.")
        return retriever

    @classmethod
    def load_retriever(cls):
        """Carga el retriever en memoria si no está cargado todavía."""
        if cls._retriever is None:
            if not os.path.exists(INDEX_DIR):
                cls._retriever = cls.populate_database()
            else:
                logger.info(f"Cargando retriever desde '{INDEX_DIR}'...")
                cls._retriever = bm25s.BM25.load(INDEX_DIR, load_corpus=True)
        return cls._retriever

    @classmethod
    def search(cls, query, k=5):
        """
        Realiza una búsqueda en el índice.

        Parámetros:
            query (str): Texto a buscar.
            k (int): Número de documentos a devolver (por defecto 5).

        Retorna:
            list[tuple[str, float]]: Lista de (documento, score)
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
