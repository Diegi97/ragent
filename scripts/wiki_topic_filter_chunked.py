"""
Chunked topic filtering for the English Wikipedia dataset.

This script mirrors the logic of wiki_topic_filter.py but processes the dataset
in chunks to avoid loading all documents in memory at once.
"""

import argparse
import logging
from pathlib import Path
import random
from typing import Iterable, List

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ragent_core.config.logging import configure_logging

DEFAULT_DATASET_NAME = "wikimedia/wikipedia"
DEFAULT_DATASET_CONFIG = "20231101.en"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_PREFIX_CHARS = 750
DEFAULT_BATCH_SIZE = 128
DEFAULT_MODEL_NAME = "MongoDB/mdbr-leaf-mt"
DEFAULT_CHUNK_SIZE = 250_000

TOPICS = {
    "space_exploration": "This article explains the history, technology, and science behind space missions. It describes how rockets work, the challenges of human spaceflight, specific missions to the Moon or Mars, the design of spacecraft, and discoveries made by space agencies like NASA. The text discusses orbital mechanics, propulsion systems, life support, or the experiences of astronauts during missions.",
    "marine_biology": "This article describes underwater ecosystems, the biology and behavior of ocean creatures, or marine research. It explains how fish, whales, sharks, or coral reefs function, discusses the ecology of deep sea habitats, or examines scientific studies of aquatic life. The text covers topics like marine food chains, ocean conservation efforts, or the physiology of specific sea animals.",
    "mythology": "This article tells the stories of gods, heroes, and mythical creatures from ancient cultures. It explains the meaning behind myths, describes legendary figures and their adventures, or analyzes folklore traditions. The text discusses specific mythological narratives from Greek, Norse, Egyptian, or other traditions, explaining their cultural significance and the tales themselves.",
    "ancient_civilizations": "This article describes the history, culture, and achievements of ancient societies. It explains how civilizations like Egypt, Rome, Greece, or Mesopotamia rose and fell, discusses their architecture, governance, and daily life, or examines archaeological discoveries. The text covers specific rulers, battles, cultural practices, or technological innovations of the ancient world.",
    "video_games": "This article describes a specific video game, its gameplay mechanics, development history, or cultural impact. It explains how the game was created, discusses its story and design choices, or reviews its reception. The text covers the history of game studios, the evolution of specific franchises, or analyzes game design principles and player experiences.",
    "nba": "This article describes NBA basketball history, specific players, teams, or memorable games. It explains career achievements, playing styles, championship runs, or franchise histories. The text discusses individual players' biographies, coaching strategies, historic rivalries, or analyzes specific seasons and playoff series in professional basketball.",
    "fantasy_scifi": "This article describes a fantasy or science fiction story, its world-building, characters, and themes. It explains the plot of a novel or series, discusses the author's creative vision, or analyzes themes like magic systems, futuristic technology, or alternate worlds. The text covers specific books, their narrative structure, and their place in the genre's history.",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        required=True,
        choices=list(TOPICS.keys()),
        help="Topic name to process.",
    )
    parser.add_argument(
        "--sim",
        type=float,
        default=0.6,
        help="Keep documents with similarity above this threshold.",
    )
    parser.add_argument(
        "--max-docs", type=int, default=0, help="Limit total docs processed."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of docs to process per chunk.",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save filtered docs with embeddings per chunk.",
    )
    parser.add_argument(
        "--load-embeddings",
        action="store_true",
        help="Reuse saved per-chunk embeddings when available.",
    )
    parser.add_argument(
        "--output-dir", default="data/wikipedia_topics", help="Output directory."
    )
    parser.add_argument(
        "--embeddings-dir",
        default="data/wikipedia_embeddings",
        help="Output directory for topic-agnostic embeddings.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chunks that already have outputs.",
    )
    return parser.parse_args()


def _get_doc_id(ex: dict) -> str:
    for key in ("id", "pageid", "wiki_id"):
        if key in ex and ex[key] is not None:
            return str(ex[key])
    return ""


def _iter_selected_records(paths: List[Path]) -> Iterable[dict]:
    for path in paths:
        df = pd.read_parquet(path)
        for rec in df.to_dict("records"):
            yield rec


def _chunk_paths(chunk_dir: Path, chunk_idx: int) -> dict:
    return {
        "embeddings_parquet": chunk_dir / f"chunk_{chunk_idx:05d}_embeddings.parquet",
        "selected_parquet": chunk_dir / f"chunk_{chunk_idx:05d}_selected.parquet",
        "index_parquet": chunk_dir / f"chunk_{chunk_idx:05d}_index.parquet",
        "stats_json": chunk_dir / f"chunk_{chunk_idx:05d}_stats.json",
    }


def main() -> None:
    args = _parse_args()
    configure_logging("INFO")
    logger = logging.getLogger(__name__)

    if args.sim is None:
        raise ValueError("--sim is required for chunked processing.")

    topic = args.topic
    topic_query = TOPICS[topic]

    base_output_dir = Path(args.output_dir) / topic
    chunk_dir = base_output_dir / "chunks"
    final_dir = base_output_dir / "final"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    embeddings_base_dir = Path(args.embeddings_dir) / DEFAULT_DATASET_CONFIG
    embeddings_chunk_dir = embeddings_base_dir / "chunks"
    embeddings_chunk_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model %s", DEFAULT_MODEL_NAME)
    model = SentenceTransformer(DEFAULT_MODEL_NAME)

    logger.info("Encoding topic prompt for '%s'", topic)
    topic_emb = model.encode([topic_query], prompt_name="query")[0]

    logger.info(
        "Loading dataset metadata %s (%s)",
        DEFAULT_DATASET_NAME,
        DEFAULT_DATASET_CONFIG,
    )
    ds_meta = load_dataset(
        DEFAULT_DATASET_NAME, DEFAULT_DATASET_CONFIG, split=DEFAULT_SPLIT
    )
    total_docs = len(ds_meta)
    if args.max_docs and args.max_docs < total_docs:
        total_docs = args.max_docs
    logger.info("Total docs to process: %d", total_docs)

    chunk_size = args.chunk_size
    num_chunks = (total_docs + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_docs)
        paths = _chunk_paths(chunk_dir, chunk_idx)
        embedding_paths = _chunk_paths(embeddings_chunk_dir, chunk_idx)

        if args.resume and paths["selected_parquet"].exists():
            logger.info("Skipping chunk %d (output exists)", chunk_idx)
            continue

        docs = []
        metas = []
        doc_embs = None

        if args.load_embeddings and embedding_paths["embeddings_parquet"].exists():
            logger.info("Loading embeddings for chunk %d", chunk_idx)
            df = pd.read_parquet(embedding_paths["embeddings_parquet"])
            metas = df[["id", "title", "text"]].to_dict("records")
            doc_embs = np.stack(df["embedding"].values)
            logger.info("Loaded %d embeddings", len(metas))
        else:
            logger.info("Loading chunk %d: docs %d:%d", chunk_idx, start, end)
            ds_chunk = load_dataset(
                DEFAULT_DATASET_NAME,
                DEFAULT_DATASET_CONFIG,
                split=f"{DEFAULT_SPLIT}[{start}:{end}]",
            )

            if args.max_docs and end - start < chunk_size:
                logger.info("Final chunk capped by --max-docs at %d docs", end - start)

            logger.info("Filtering %d documents by non-empty text", len(ds_chunk))
            for ex in tqdm(ds_chunk, desc=f"Filtering chunk {chunk_idx}"):
                text = (ex.get("text") or "").strip()
                if not text:
                    continue
                title = (ex.get("title") or "").strip()
                snippet = text[:DEFAULT_TEXT_PREFIX_CHARS]
                docs.append(f"title: {title or 'none'} | text: {snippet}")
                metas.append({"id": _get_doc_id(ex), "title": title, "text": text})

            logger.info(
                "Chunk %d kept %d documents after filtering",
                chunk_idx,
                len(docs),
            )

            if docs:
                logger.info("Encoding %d documents", len(docs))
                doc_embs = model.encode(
                    docs, show_progress_bar=True, batch_size=DEFAULT_BATCH_SIZE
                )

                if args.save_embeddings:
                    df = pd.DataFrame(metas)
                    df["embedding"] = list(doc_embs)
                    df.to_parquet(embedding_paths["embeddings_parquet"])
            else:
                doc_embs = np.array([])

        if not metas:
            logger.info(
                "Chunk %d has no docs after filtering; writing empty outputs",
                chunk_idx,
            )
            pd.DataFrame(columns=["id", "title", "text", "topic", "topic_score"]).to_parquet(
                paths["selected_parquet"]
            )
            pd.DataFrame(columns=["id", "topic_score"]).to_parquet(paths["index_parquet"])
            continue

        topic_scores = model.similarity(topic_emb, doc_embs)[0].numpy()
        selected_indices = np.where(topic_scores >= args.sim)[0]
        logger.info(
            "Chunk %d found %d documents with similarity >= %.4f",
            chunk_idx,
            len(selected_indices),
            args.sim,
        )

        selected_records = []
        index_records = []
        for idx in selected_indices:
            score = float(topic_scores[idx])
            meta = metas[idx]
            selected_records.append(
                {
                    "id": meta["id"],
                    "title": meta["title"],
                    "text": meta["text"],
                    "topic": topic,
                    "topic_score": score,
                }
            )
            index_records.append({"id": meta["id"], "topic_score": score})

        pd.DataFrame(selected_records).to_parquet(paths["selected_parquet"])
        pd.DataFrame(index_records).to_parquet(paths["index_parquet"])

        stats = {
            "chunk_idx": chunk_idx,
            "start": start,
            "end": end,
            "docs_loaded": len(metas),
            "docs_filtered": len(metas),
            "docs_selected": len(selected_records),
        }
        pd.Series(stats).to_json(paths["stats_json"])

    logger.info("Building final dataset from chunk outputs")
    selected_paths = sorted(chunk_dir.glob("chunk_*_selected.parquet"))
    if not selected_paths:
        raise FileNotFoundError("No chunk outputs found; nothing to assemble.")

    out_ds = Dataset.from_generator(
        _iter_selected_records, gen_kwargs={"paths": selected_paths}
    )
    final_dir.mkdir(parents=True, exist_ok=True)
    out_ds.save_to_disk(str(final_dir))
    logger.info("Saved dataset to %s", final_dir)


if __name__ == "__main__":
    main()
