import argparse
import logging
from pathlib import Path

from pydantic import (
    ValidationError,
)

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    validate_question_rubric_file,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation.workspace import (
    load_workspace_doc_ids,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a PI-generated question-rubric Markdown file."
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        allowed_doc_ids = load_workspace_doc_ids(Path.cwd() / "facts")
        validate_question_rubric_file(args.path, allowed_doc_ids=allowed_doc_ids)
    except (OSError, ValueError, ValidationError) as exc:
        logger.error("Invalid question-rubric file: %s", exc)
        return 1
    logger.info("Question-rubric file is valid: %s", args.path)
    return 0
