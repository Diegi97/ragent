"""Prepare a pinned snapshot of Harvey's synthetic firm, without benchmark tasks."""

import argparse
import hashlib
import json
import logging
import re
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from datasets import Dataset, Features, Value

from . import DATA_DIR, DESCRIPTION
from .extract import SUPPORTED_EXTENSIONS, extract_document

logger = logging.getLogger(__name__)
REPO_URL = "https://github.com/harveyai/harvey-labs.git"
REVISION = "c2488cfa24fd01ee88016a121479a2f86b394bd4"
DMS_PATH = "tasks/firm-knowledge/dms"


def _git(repository: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def download_repository(repository: Path, revision: str) -> None:
    """Sparse checkout avoids downloading the rest of the large LAB repository."""
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "remote", "add", "origin", REPO_URL)
    _git(repository, "sparse-checkout", "init", "--cone")
    _git(repository, "sparse-checkout", "set", DMS_PATH)
    logger.info("Downloading Harvey firm documents at %s", revision)
    _git(repository, "fetch", "--depth=1", "--filter=blob:none", "origin", revision)
    _git(repository, "checkout", "--detach", "FETCH_HEAD")


def tracked_documents(repository: Path, revision: str) -> list[tuple[str, str]]:
    """Return only regular DMS matter files and their expected Git blob hashes."""
    if _git(repository, "rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError("Source checkout does not match the requested revision")
    entries = _git(repository, "ls-tree", "-rz", revision, "--", DMS_PATH)
    documents = []
    prefix = f"{DMS_PATH}/"
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, blob_sha = metadata.decode().split()
        path = raw_path.decode("utf-8")
        relative = PurePosixPath(path).relative_to(DMS_PATH)
        if (
            mode != "100644"
            or kind != "blob"
            or not path.startswith(prefix)
            or ".." in relative.parts
            or len(relative.parts) < 3
            or relative.parts[0] != "matters"
            or relative.suffix.lower() not in SUPPORTED_EXTENSIONS
        ):
            raise ValueError(f"Unexpected file in Harvey's DMS: {path}")
        documents.append((relative.as_posix(), blob_sha))
    if not documents:
        raise ValueError("No Harvey firm documents found in the checkout")
    return sorted(documents)


def document_id(relative_path: str) -> int:
    """Stable positive int64 IDs, independent of file ordering or corpus subsets."""
    digest = hashlib.sha256(f"{DMS_PATH}/{relative_path}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1


def normalize_document(relative_path: str, content: str, revision: str) -> dict:
    path = PurePosixPath(relative_path)
    matter_id = path.parts[1]
    title = f"{matter_id} / {path.stem.replace('-', ' ').replace('_', ' ')}"
    source = (
        f"https://github.com/harveyai/harvey-labs/blob/{revision}/"
        f"{quote(DMS_PATH + '/' + relative_path, safe='/')}"
    )
    return {
        "id": document_id(relative_path),
        "title": title,
        "text": (
            f"# {title}\n\n"
            "Firm: Calderwood & Harkness (synthetic)\n"
            f"Matter: {matter_id}\n"
            f"Path: {relative_path}\n"
            f"Source: {source}\n\n{content}"
        ),
    }


def prepare_dataset(
    output_dir: Path = DATA_DIR,
    *,
    revision: str = REVISION,
    source_dir: Path | None = None,
) -> Path:
    """Keep a complete raw and normalized snapshot; never publish partial output."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full 40-character Git commit SHA")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; use a new --output-dir")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        temporary_path = Path(temporary)
        repository = (
            Path(source_dir)
            if source_dir is not None
            else temporary_path / "repository"
        )
        if source_dir is None:
            download_repository(repository, revision)
        documents = tracked_documents(repository, revision)
        staging = temporary_path / "snapshot"
        staging.mkdir()
        # Retain the repository's attribution alongside the downloaded documents.
        (staging / "LICENSE").write_bytes(
            _git(repository, "show", f"{revision}:LICENSE")
        )
        counts: Counter[str] = Counter()
        matters: set[str] = set()
        seen_ids: set[int] = set()
        lengths = []
        total_bytes = 0
        with (
            (staging / "train.jsonl").open("w", encoding="utf-8") as normalized,
            (staging / "documents.jsonl").open("w", encoding="utf-8") as metadata,
        ):
            for index, (relative_path, blob_sha) in enumerate(documents, start=1):
                path = repository / DMS_PATH / relative_path
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Missing or non-regular source document: {path}")
                raw = path.read_bytes()
                actual_sha = hashlib.sha1(
                    f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False
                ).hexdigest()
                if actual_sha != blob_sha:
                    raise ValueError(
                        f"Source document differs from pinned revision: {path}"
                    )
                raw_path = staging / "raw" / relative_path
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(raw)
                try:
                    content = extract_document(raw_path)
                except Exception as exc:
                    raise ValueError(f"Cannot extract {relative_path}: {exc}") from exc
                row = normalize_document(relative_path, content, revision)
                if row["id"] in seen_ids:
                    raise ValueError(f"Document ID collision for {relative_path}")
                seen_ids.add(row["id"])
                normalized.write(json.dumps(row, ensure_ascii=False) + "\n")
                word_count = len(row["text"].split())
                matter = PurePosixPath(relative_path).parts[1]
                metadata.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "path": relative_path,
                            "matter_id": matter,
                            "git_blob_sha": blob_sha,
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "bytes": len(raw),
                            "word_count": word_count,
                        }
                    )
                    + "\n"
                )
                counts[path.suffix.lower()] += 1
                matters.add(matter)
                lengths.append(word_count)
                total_bytes += len(raw)
                if index % 250 == 0 or index == len(documents):
                    logger.info(
                        "Extracted %s/%s Harvey documents", index, len(documents)
                    )
        dataset = Dataset.from_json(
            str(staging / "train.jsonl"),
            features=Features(
                {
                    "id": Value("int64"),
                    "title": Value("string"),
                    "text": Value("string"),
                }
            ),
            cache_dir=str(temporary_path / "arrow_cache"),
        )
        if len(dataset) != len(documents):
            raise ValueError("Normalized dataset count does not match the source tree")
        dataset.info.description = DESCRIPTION
        dataset.info.dataset_name = "harvey_synthetic_firm"
        dataset.save_to_disk(str(staging / "dataset"))
        manifest = {
            "repository": REPO_URL,
            "revision": revision,
            "source_directory": DMS_PATH,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "documents": len(documents),
            "matters": len(matters),
            "formats": dict(sorted(counts.items())),
            "source_bytes": total_bytes,
            "word_count": {
                "min": min(lengths),
                "max": max(lengths),
                "total": sum(lengths),
            },
            "documents_over_8000_words": sum(length > 8000 for length in lengths),
            "length_filter_applied": False,
            "id_scheme": "SHA-256 of repository-relative path, first 8 bytes masked to 63 bits; zero mapped to one",
            "extraction": "Text and tables; no OCR, image transcription, or formula recalculation",
            "schema_version": 1,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    logger.info("Saved %s Harvey documents to %s", len(documents), output_dir)
    return output_dir / "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--revision", default=REVISION, help="Full Git commit SHA")
    parser.add_argument(
        "--source-dir", type=Path, help="Reuse a checkout at the requested revision"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prepare_dataset(args.output_dir, revision=args.revision, source_dir=args.source_dir)


if __name__ == "__main__":
    main()
