import logging
from functools import lru_cache
from typing import List, Sequence

from chonkie import RecursiveChunker
from tqdm.auto import tqdm

from ragent_core.retrievers.document import Document

logger = logging.getLogger(__name__)

DOCUMENT_ID_KEY = "document_id"
TOKENIZER_NAME = "word"


@lru_cache(maxsize=32)
def _get_chunker(chunk_size_tokens: int) -> RecursiveChunker:
    return RecursiveChunker(chunk_size=chunk_size_tokens, tokenizer=TOKENIZER_NAME)


def chunk_documents(
    documents: Sequence[Document],
    chunk_size_tokens: int,
) -> List[Document]:
    """Split source documents into chunk documents.

    Each chunk has its own ``id`` plus a top-level ``document_id`` (also in
    ``metadata[DOCUMENT_ID_KEY]``) that points back to the source document's
    id. Full documents live in a separate table (see
    :meth:`LanceDBRetriever.build_index`).
    """
    chunks: List[Document] = []
    next_chunk_id = 0
    chunker = _get_chunker(chunk_size_tokens)
    for doc in tqdm(documents, desc="Chunking documents", unit="doc"):
        for chunk in chunker.chunk(doc.content):
            chunks.append(
                Document(
                    id=next_chunk_id,
                    title=doc.title,
                    content=chunk.text,
                    metadata={
                        **doc.metadata,
                        DOCUMENT_ID_KEY: doc.id,
                    },
                    document_id=doc.id,
                )
            )
            next_chunk_id += 1
    logger.info("Chunked %d documents into %d chunks.", len(documents), len(chunks))
    return chunks
