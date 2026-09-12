import argparse
import asyncio
import json
import sys
from pathlib import Path

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.solver import (
    SolverRuntime,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    return parser.parse_args()


def main() -> None:
    try:
        args = _parse_args()
        runtime = SolverRuntime.from_environment()
        print(
            json.dumps(asyncio.run(runtime.solve(args.candidate)), ensure_ascii=False)
        )
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
