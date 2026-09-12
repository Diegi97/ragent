import argparse
import asyncio
import logging
from pathlib import Path

from ragent_core.judges.evaluation.display import print_summary
from ragent_core.judges.evaluation.experiments import run_experiments
from ragent_core.judges.evaluation.inputs import load_ground_truth, resolve_traces_path
from ragent_core.judges.evaluation.metrics import build_metrics
from ragent_core.judges.evaluation.output import (
    require_output_available,
    write_json,
    write_jsonl,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.pinference.ai/api/v1"


DEFAULT_API_KEY_VAR = "PRIME_API_KEY"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ground_truth_path",
        type=Path,
        help="Teacher-labeled Verifiers traces.jsonl file or its run directory.",
    )
    parser.add_argument("judge_model")
    parser.add_argument(
        "--criteria-per-call",
        type=_positive_int,
        nargs="+",
        required=True,
        help="One or more maximum rubric batch sizes to compare.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-var", default=DEFAULT_API_KEY_VAR)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=_positive_int, default=4096)
    parser.add_argument(
        "--max-concurrent",
        type=_positive_int,
        default=16,
        help="Maximum in-flight judge calls across examples.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Metrics JSON path. Defaults next to the input file.",
    )
    parser.add_argument(
        "--judgments-output",
        type=Path,
        help="Criterion-level JSONL path. Defaults next to the metrics file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing metrics and judgment files.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    criteria_per_call = list(dict.fromkeys(args.criteria_per_call))
    ground_truth_path = resolve_traces_path(args.ground_truth_path)
    metrics_output = (
        args.metrics_output.expanduser().resolve()
        if args.metrics_output
        else ground_truth_path.with_name(f"{ground_truth_path.stem}.metrics.json")
    )
    judgments_output = (
        args.judgments_output.expanduser().resolve()
        if args.judgments_output
        else metrics_output.with_name(f"{metrics_output.stem}.judgments.jsonl")
    )
    if metrics_output == judgments_output:
        raise ValueError(
            "--metrics-output and --judgments-output must be different files"
        )
    for path in (metrics_output, judgments_output):
        require_output_available(path, args.overwrite)

    examples = load_ground_truth(ground_truth_path)
    rows, calls = asyncio.run(
        run_experiments(
            examples,
            criteria_per_call,
            judge_model=args.judge_model,
            base_url=args.base_url,
            api_key_var=args.api_key_var,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_concurrent=args.max_concurrent,
        )
    )
    metrics = build_metrics(
        examples,
        rows,
        calls,
        criteria_per_call=criteria_per_call,
        ground_truth_path=ground_truth_path,
        judge_model=args.judge_model,
        base_url=args.base_url,
        api_key_var=args.api_key_var,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
    )
    write_jsonl(judgments_output, rows, args.overwrite)
    write_json(metrics_output, metrics, args.overwrite)

    print_summary(metrics)
    logger.info("Wrote detailed metrics to %s", metrics_output)
    logger.info("Wrote criterion-level judgments to %s", judgments_output)
