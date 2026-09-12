import re
from pathlib import Path

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation.markdown import (
    parse_doc_ids,
)

DOCUMENT_HEADING_PATTERN = re.compile(r"### Documents? (.+)")


def load_workspace_doc_ids(facts_directory: Path) -> set[int]:
    doc_ids: set[int] = set()
    for path in facts_directory.rglob("*.md"):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = DOCUMENT_HEADING_PATTERN.fullmatch(line.strip())
            if match is None:
                continue
            doc_ids.update(
                parse_doc_ids(
                    match.group(1),
                    field_name=f"document heading in {path}",
                )
            )
    if not doc_ids:
        raise ValueError(f"no document headings found in {facts_directory}")
    return doc_ids


def document_heading(doc_ids: tuple[int, ...]) -> str:
    label = "Document" if len(doc_ids) == 1 else "Documents"
    return f"### {label} {', '.join(map(str, doc_ids))}"
