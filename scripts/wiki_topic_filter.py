"""
This script is used to create new data sources based on specific topics of the English Wikipedia.
It filters articles from the English Wikipedia dataset, ranks them by topic relevance using
embeddings, and outputs a curated dataset for use as a data source.
"""

import argparse
import logging
from pathlib import Path
import random
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ragent_core.config.logging import configure_logging

DEFAULT_DATASET_NAME = "wikimedia/wikipedia"
DEFAULT_DATASET_CONFIG = "20231101.en"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_PREFIX_CHARS = 1500
DEFAULT_BATCH_SIZE = 128
DEFAULT_MODEL_NAME = "MongoDB/mdbr-leaf-mt"

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
        "--top-k", type=int, default=5000, help="Top documents for the topic."
    )
    parser.add_argument(
        "--max-words", type=int, default=8000, help="Max words per document."
    )
    parser.add_argument(
        "--max-docs", type=int, default=0, help="Limit total docs processed."
    )
    parser.add_argument(
        "--output-dir", default="data/wikipedia_topics", help="Output directory."
    )
    parser.add_argument(
        "--load-parquet",
        default="",
        help="Path to parquet with pre-filtered docs and embeddings.",
    )
    parser.add_argument(
        "--save-parquet",
        action="store_true",
        help="Save filtered docs with embeddings as parquet.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default="", help="HF Hub repo id.")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def _get_doc_id(ex: dict) -> str:
    for key in ("id", "pageid", "wiki_id"):
        if key in ex and ex[key] is not None:
            return str(ex[key])
    return ""


def main() -> None:
    args = _parse_args()
    configure_logging("INFO")
    logger = logging.getLogger(__name__)

    topic = args.topic
    topic_query = TOPICS[topic]

    logger.info("Loading model %s", DEFAULT_MODEL_NAME)
    model = SentenceTransformer(DEFAULT_MODEL_NAME)

    logger.info("Encoding topic prompt for '%s'", topic)
    topic_emb = model.encode([topic_query], prompt_name="query")[0]

    if args.load_parquet:
        # Load pre-filtered documents with embeddings from parquet

        parquet_path = Path(args.load_parquet)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        logger.info("Loading documents from %s", parquet_path)
        df = pd.read_parquet(parquet_path)
        metas = df[["id", "title", "text"]].to_dict("records")
        doc_embs = np.stack(df["embedding"].values)
        logger.info("Loaded %d documents with embeddings", len(metas))
    else:
        # Load and filter from Wikipedia dataset
        logger.info(
            "Loading dataset %s (%s)", DEFAULT_DATASET_NAME, DEFAULT_DATASET_CONFIG
        )
        ds = load_dataset(
            DEFAULT_DATASET_NAME, DEFAULT_DATASET_CONFIG, split=DEFAULT_SPLIT
        )
        if args.max_docs and args.max_docs < len(ds):
            indices = random.sample(range(len(ds)), args.max_docs)
            ds = ds.select(indices)

        logger.info(
            "Filtering %d documents by word count (<=%d)", len(ds), args.max_words
        )
        docs, metas = [], []
        for ex in tqdm(ds, desc="Filtering"):
            text = (ex.get("text") or "").strip()
            if not text or len(text.split()) > args.max_words:
                continue
            title = (ex.get("title") or "").strip()
            snippet = text[:DEFAULT_TEXT_PREFIX_CHARS]
            docs.append(f"title: {title or 'none'} | text: {snippet}")
            metas.append({"id": _get_doc_id(ex), "title": title, "text": text})
        logger.info("Kept %d documents after filtering", len(docs))

        logger.info("Encoding %d documents", len(docs))
        doc_embs = model.encode(
            docs, show_progress_bar=True, batch_size=DEFAULT_BATCH_SIZE
        )

        # Save filtered docs with embeddings to parquet for reuse
        if args.save_parquet:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = output_dir / "filtered_docs.parquet"
            df = pd.DataFrame(metas)
            df["embedding"] = list(doc_embs)
            df.to_parquet(parquet_path)
            logger.info(
                "Saved %d documents with embeddings to %s", len(metas), parquet_path
            )

    # Compute topic scores and select top-k
    topic_scores = model.similarity(topic_emb, doc_embs)[0].numpy()
    top_indices = np.argsort(topic_scores)[::-1][: args.top_k]
    records = []
    for idx in top_indices:
        records.append(
            {
                "id": metas[idx]["id"],
                "title": metas[idx]["title"],
                "text": metas[idx]["text"],
                "topic": topic,
                "topic_score": float(topic_scores[idx]),
            }
        )

    logger.info("Creating dataset with %d rows", len(records))
    out_ds = Dataset.from_list(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_ds.save_to_disk(str(output_dir))
    logger.info("Saved dataset to %s", output_dir)

    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--repo-id required with --push-to-hub")
        logger.info("Pushing to hub: %s", args.repo_id)
        out_ds.push_to_hub(args.repo_id, private=args.private)


if __name__ == "__main__":
    main()
