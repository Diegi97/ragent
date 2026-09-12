"""Shared extraction and locked publication of local repository corpora."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetInfo, load_from_disk
from filelock import FileLock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepositoryCorpus:
    name: str
    repository_url: str
    content_directory: str
    extract: Callable[[Path], list[dict[str, Any]]]

    @property
    def directory(self) -> Path:
        return Path("data") / self.name

    def load(self) -> Dataset:
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        with self._lock():
            if not (self.directory / "dataset").exists():
                self._publish()
            dataset = load_from_disk(str(self.directory / "dataset"))
            if not isinstance(dataset, Dataset):
                raise TypeError(f"Expected Dataset in {self.directory}")
            return dataset

    def prepare(self) -> None:
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        with self._lock():
            self._publish()

    def _lock(self) -> FileLock:
        # Keep the lock outside the replaced directory; readers share it too.
        return FileLock(self.directory.parent / f".{self.name}.lock")

    def _publish(self) -> None:
        with tempfile.TemporaryDirectory() as checkout:
            logger.info("Cloning %s", self.repository_url)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repository_url, checkout],
                check=True,
            )
            content = find_content_directory(Path(checkout), self.content_directory)
            records = self.extract(content)
        if not records:
            raise ValueError(f"No source documents were extracted for {self.name}")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{self.name}-", dir=self.directory.parent)
        )
        backup = self.directory.with_name(f".{self.name}.backup-{uuid.uuid4().hex}")
        try:
            dataset = Dataset.from_list(
                records, info=DatasetInfo(dataset_name=self.name)
            )
            dataset.save_to_disk(str(staging / "dataset"))
            with (staging / "train.jsonl").open("w", encoding="utf-8") as output:
                for record in records:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            # Non-empty directories cannot be replaced directly. Managed readers
            # hold the same lock while loading, and failed swaps restore the cache.
            if self.directory.exists():
                os.replace(self.directory, backup)
            try:
                os.replace(staging, self.directory)
            except BaseException:
                if backup.exists():
                    os.replace(backup, self.directory)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        logger.info("Dataset saved to %s", self.directory)


def find_content_directory(checkout: Path, name: str) -> Path:
    preferred = checkout / name
    if preferred.is_dir():
        return preferred
    for root, directories, _ in os.walk(checkout):
        if name in directories:
            return Path(root) / name
    raise FileNotFoundError(f"Could not locate {name} directory in cloned repository")


def extract_markdown_files(
    content_directory: Path, files: Iterable[Path]
) -> list[dict[str, Any]]:
    records = []
    for document_id, path in enumerate(files):
        content = path.read_text(encoding="utf-8")
        title = next(
            (
                line[2:].strip()
                for line in content.splitlines()
                if line.startswith("# ")
            ),
            path.stem.replace("-", " ").replace("_", " ").capitalize(),
        )
        records.append(
            {
                "id": document_id,
                "title": title,
                "text": content,
                "path": str(path.relative_to(content_directory)),
            }
        )
    return records
