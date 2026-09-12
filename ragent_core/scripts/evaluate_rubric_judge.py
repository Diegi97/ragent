"""Evaluate batched rubric judgments against criterion-level teacher labels.

First run the deep-search evaluation configured by
``environments/ragent_deep_search/rubric_judge_ground_truth_generation.toml``.
It uses the null harness to generate retrieval-assisted answers and configures the
teacher judge with ``max_criteria = 1``, producing one independent teacher call per
rubric criterion. The resulting Verifiers ``traces.jsonl`` is the ground-truth input
to this script; no intermediate conversion or ground-truth generation script is
needed. A run output directory may be passed in place of the traces file.

For every requested ``--criteria-per-call`` value, this script rejudges the saved
answers with the repository's production ``RubricJudge`` prompt and parser. It writes
detailed metrics and criterion-level predictions while printing only a compact
aggregate summary. Malformed or incomplete judge calls remain experimental failures
and count as incorrect in strict accuracy.

Example:
    uv run --env-file environments/ragent_deep_search/.env         --project ragent_core python         ragent_core/scripts/evaluate_rubric_judge.py         environments/ragent_deep_search/outputs/<run>/traces.jsonl         openai/gpt-5.4-nano --criteria-per-call 1 2 4 8
"""

from ragent_core.judges.evaluation.cli import main

if __name__ == "__main__":
    main()
