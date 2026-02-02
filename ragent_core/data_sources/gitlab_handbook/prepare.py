import os
import subprocess
import tempfile
import logging
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetInfo

logger = logging.getLogger(__name__)


def clone_repo(repo_url, temp_dir):
    logger.info(f"Cloning {repo_url} into {temp_dir}")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, temp_dir], check=True)


def extract_markdown(content_dir):
    data = []
    content_path = Path(content_dir)
    doc_id = 0

    for md_file in content_path.rglob("*.md"):
        if md_file.name.startswith("_index."):
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
            # Simple title extraction: first H1 or filename
            title = md_file.stem.replace("-", " ").replace("_", " ").capitalize()
            for line in content.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            # Get relative path for Source reference
            rel_path = md_file.relative_to(content_path)

            data.append(
                {"id": doc_id, "title": title, "text": content, "path": str(rel_path)}
            )
            doc_id += 1
        except Exception as e:
            logger.error(f"Error reading {md_file}: {e}")

    return data


def prepare_dataset():
    repo_url = "https://gitlab.com/gitlab-com/content-sites/handbook.git"
    output_path = Path("data/gitlab_handbook")
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        clone_repo(repo_url, temp_dir)
        content_dir = os.path.join(temp_dir, "content")

        if not os.path.exists(content_dir):
            # Try to find content dir if it's nested
            logger.warning(
                f"Content directory not found at {content_dir}, searching..."
            )
            for root, dirs, files in os.walk(temp_dir):
                if "content" in dirs:
                    content_dir = os.path.join(root, "content")
                    break

        logger.info(f"Extracting markdown from {content_dir}")
        data = extract_markdown(content_dir)

        logger.info(f"Extracted {len(data)} documents")

        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(
            df, info=DatasetInfo(dataset_name="gitlab_handbook")
        )

        dataset_path = output_path / "dataset"
        dataset.save_to_disk(str(dataset_path))

        # Also save as jsonl for easier inspection/loading via load_dataset
        jsonl_path = output_path / "train.jsonl"
        df.to_json(jsonl_path, orient="records", lines=True)

        logger.info(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_dataset()
