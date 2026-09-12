import logging
import re
from pathlib import Path

from ragent_core.data_sources.repository import RepositoryCorpus


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
    document_records = []
    content_path = Path(content_dir)
    doc_id = 0

    for mdx_file in content_path.rglob("*.mdx"):
        if mdx_file.name.startswith("_index."):
            continue

        content = mdx_file.read_text(encoding="utf-8")
        title = _extract_title(content, mdx_file.stem)
        rel_path = mdx_file.relative_to(content_path)
        document_records.append(
            {
                "id": doc_id,
                "title": title,
                "text": content,
                "path": str(rel_path),
            }
        )
        doc_id += 1

    return document_records


corpus = RepositoryCorpus(
    "posthog_com", "https://github.com/PostHog/posthog.com.git", "contents", extract_mdx
)


def prepare_dataset() -> None:
    """Explicitly rebuild this source cache under its shared publication lock."""
    corpus.prepare()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_dataset()
