import logging
from typing import Any, List, Optional, Sequence

from datasets import Dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def prepare_chunk_docs(
    chunk_dataset: Optional[Dataset],
    dataset: Dataset,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> List[dict[str, Any]]:
    if chunk_dataset is not None:
        return chunk_dataset.to_list()

    logger.info("Building in-memory BM25 chunk index...")
    chunk_docs = chunk_documents(
        dataset=dataset,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    logger.info(
        "Chunked %d documents into %d chunks.",
        len(dataset),
        len(chunk_docs),
    )
    return chunk_docs


def chunk_documents(
    dataset: Dataset,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> List[dict[str, Any]]:
    chunk_docs: List[dict[str, Any]] = []
    chunk_id = 0
    for doc_index, doc in enumerate(tqdm(dataset, desc="Chunking documents", unit="doc")):
        doc_id = doc.get("id", doc_index)
        title = doc.get("title", "")
        text = doc.get("text", "")
        for chunk_text in split_text(
            text=text,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        ):
            chunk_docs.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "title": title,
                    "doc_id": doc_id,
                }
            )
            chunk_id += 1
    return chunk_docs


def split_text(
    text: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> List[str]:
    if not text:
        return [text]
    chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="o200k_base",
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
    )
    chunks = chunker.split_text(text)
    return chunks or [text]
