import logging
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetInfo

logger = logging.getLogger(__name__)


def clone_repo(repo_url: str, temp_dir: str) -> None:
    logger.info("Cloning %s into %s", repo_url, temp_dir)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, temp_dir], check=True)


def find_cheatsheets_dir(root_dir: str) -> str:
    default_path = os.path.join(root_dir, "cheatsheets")
    if os.path.isdir(default_path):
        return default_path

    for root, dirs, _files in os.walk(root_dir):
        if "cheatsheets" in dirs:
            return os.path.join(root, "cheatsheets")

    raise FileNotFoundError("Could not locate cheatsheets directory in cloned repo")


def extract_markdown(content_dir: str) -> list[dict]:
    data = []
    content_path = Path(content_dir)
    doc_id = 0

    for md_file in sorted(content_path.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")

            title = md_file.stem.replace("-", " ").replace("_", " ").capitalize()
            for line in content.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            rel_path = md_file.relative_to(content_path)

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
            logger.error("Error reading %s: %s", md_file, exc)

    return data


def prepare_dataset() -> None:
    repo_url = "https://github.com/OWASP/CheatSheetSeries.git"
    output_path = Path("data/owasp_cheatsheets")
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        clone_repo(repo_url, temp_dir)
        content_dir = find_cheatsheets_dir(temp_dir)

        logger.info("Extracting markdown from %s", content_dir)
        data = extract_markdown(content_dir)
        logger.info("Extracted %d documents", len(data))

        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(
            df, info=DatasetInfo(dataset_name="owasp_cheatsheets")
        )

        dataset_path = output_path / "dataset"
        dataset.save_to_disk(str(dataset_path))

        jsonl_path = output_path / "train.jsonl"
        df.to_json(jsonl_path, orient="records", lines=True)

        logger.info("Dataset saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_dataset()
