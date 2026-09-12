"""Upload generated question-rubric records to a private Hugging Face dataset.

The input batch is reproducibly divided between the dataset's standard
``train`` and ``test`` splits. Existing split assignments are preserved. Pass
``--replace-data`` to replace records from one data source instead of appending
another batch.

Usage:
    uv run --project data_pipelines python \
        data_pipelines/scripts/upload_question_rubrics.py \
        path/to/question_rubrics.jsonl gitlab_handbook
"""

from data_pipelines.publishing.question_rubrics.cli import main

if __name__ == "__main__":
    main()
