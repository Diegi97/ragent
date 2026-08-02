import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetInfo

logger = logging.getLogger(__name__)


def clone_repo(repo_url: str, temp_dir: str) -> None:
    logger.info("Cloning %s into %s", repo_url, temp_dir)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, temp_dir], check=True)


def _extract_title_from_frontmatter(lines: list[str]) -> str | None:
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            return None
        if line.lower().startswith("title:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return None


def _extract_title(content: str, fallback: str) -> str:
    lines = content.splitlines()
    title = _extract_title_from_frontmatter(lines)
    if title:
        return title
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    cleaned = re.sub(r"[-_]+", " ", fallback).strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else fallback


def extract_mdx(content_dir: str) -> list[dict]:
    data = []
    content_path = Path(content_dir)
    doc_id = 0

    for mdx_file in content_path.rglob("*.mdx"):
        if mdx_file.name.startswith("_index."):
            continue

        try:
            content = mdx_file.read_text(encoding="utf-8")
            title = _extract_title(content, mdx_file.stem)
            rel_path = mdx_file.relative_to(content_path)
            data.append(
                {
                    "id": doc_id,
                    "title": title,
                    "text": content,
                    "path": str(rel_path),
                }
            )
            doc_id += 1
        except Exception as exc:
            logger.error("Error reading %s: %s", mdx_file, exc)

    return data


def prepare_dataset() -> None:
    repo_url = "https://github.com/PostHog/posthog.com.git"
    output_path = Path("data/posthog_com")
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        clone_repo(repo_url, temp_dir)
        content_dir = os.path.join(temp_dir, "contents")

        if not os.path.exists(content_dir):
            logger.warning(
                "contents directory not found at %s, searching...", content_dir
            )
            for root, dirs, _files in os.walk(temp_dir):
                if "contents" in dirs:
                    content_dir = os.path.join(root, "contents")
                    break

        logger.info("Extracting MDX from %s", content_dir)
        data = extract_mdx(content_dir)

        logger.info("Extracted %d documents", len(data))
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df, info=DatasetInfo(dataset_name="posthog_com"))

        dataset_path = output_path / "dataset"
        dataset.save_to_disk(str(dataset_path))

        jsonl_path = output_path / "train.jsonl"
        df.to_json(jsonl_path, orient="records", lines=True)

        logger.info("Dataset saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_dataset()
