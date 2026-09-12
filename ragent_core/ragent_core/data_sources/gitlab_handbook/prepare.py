import logging
from pathlib import Path

from ragent_core.data_sources.repository import RepositoryCorpus, extract_markdown_files


def extract_markdown(content_dir: Path) -> list[dict]:
    content_dir = Path(content_dir)
    return extract_markdown_files(
        content_dir,
        (
            path
            for path in content_dir.rglob("*.md")
            if not path.name.startswith("_index.")
        ),
    )


corpus = RepositoryCorpus(
    "gitlab_handbook",
    "https://gitlab.com/gitlab-com/content-sites/handbook.git",
    "content",
    extract_markdown,
)


def prepare_dataset() -> None:
    """Explicitly rebuild this source cache under its shared publication lock."""
    corpus.prepare()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_dataset()
