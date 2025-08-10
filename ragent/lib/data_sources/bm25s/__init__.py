import os
import bm25s
import logging
from datasets import load_dataset
from ragent.config import HUGGING_FACE_DATASET, HF_TOKEN

logger = logging.getLogger(__name__)

INDEX_DIR = f"data/bm25s_index/{HUGGING_FACE_DATASET}"


def populate_database(force=False):
    """
    Crea un índice BM25 a partir de un dataset de Hugging Face.
    """
    if not force and os.path.exists(INDEX_DIR):
        logger.info(f"Índice ya existe en '{INDEX_DIR}', omitiendo creación.")
        return

    logger.info("Cargando dataset...")
    dataset = load_dataset(HUGGING_FACE_DATASET, token=HF_TOKEN)

    # Ajustar el nombre de la columna según el dataset
    corpus = dataset["train"]["text"]

    logger.info("Tokenizando corpus (sin stemmer)...")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

    logger.info("Creando BM25 retriever...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    logger.info(f"Guardando índice en '{INDEX_DIR}'...")
    retriever.save(INDEX_DIR, corpus=corpus)
    logger.info("Índice creado con éxito.")


class BM25Client:
    _retriever = None  # atributo de clase

    @classmethod
    def _load_retriever(cls):
        """Carga el retriever en memoria si no está cargado todavía."""
        if cls._retriever is None:
            if not os.path.exists(INDEX_DIR):
                raise FileNotFoundError(
                    f"No se encontró el índice '{INDEX_DIR}'. Ejecuta populate_database(force=True) primero."
                )
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
        retriever = cls._load_retriever()

        query_tokens = bm25s.tokenize(query, stopwords="en")
        results, scores = retriever.retrieve(query_tokens, k=k, corpus=retriever.corpus)

        return [
            (results[0, i], scores[0, i])
            for i in range(results.shape[1])
        ]


if __name__ == "__main__":
    populate_database(force=False)

    resultados = BM25Client.search("What is the way for using a Django Serializer?", k=3)
    for rank, (doc, score) in enumerate(resultados, start=1):
        logger.info(f"Rank {rank} (score: {score:.2f}): {doc}")
