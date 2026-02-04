#!/usr/bin/env python3
"""
Analyze RL environment evaluation results from JSONL files.

This script computes metrics grouped by difficulty and data source to understand
how these factors affect agent performance.

Usage:
    python scripts/analyze_eval_results.py path/to/results.jsonl
    
The script will display:
    1. Results grouped by difficulty level (sorted by judge reward)
    2. Results grouped by data source (sorted by judge reward)
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table


@dataclass
class Metrics:
    """Container for computed metrics."""

    count: int = 0
    judge_reward_sum: float = 0.0
    format_reward_sum: float = 0.0
    general_reward_sum: float = 0.0
    num_turns_sum: float = 0.0
    total_tool_calls_sum: float = 0.0
    search_tool_calls_sum: float = 0.0
    read_tool_calls_sum: float = 0.0
    text_scan_tool_calls_sum: float = 0.0
    generation_ms_sum: float = 0.0
    total_ms_sum: float = 0.0
    errors: int = 0
    success: int = 0

    def add_sample(self, data: Dict[str, Any]) -> None:
        """Add a sample to the metrics."""
        self.count += 1
        self.judge_reward_sum += data.get("judge_reward", 0.0)
        self.format_reward_sum += data.get("format_reward", 0.0)
        self.general_reward_sum += data.get("reward", 0.0)
        self.num_turns_sum += data.get("num_turns", 0.0)
        self.total_tool_calls_sum += data.get("total_tool_calls", 0.0)
        self.search_tool_calls_sum += data.get("search_tool_calls", 0.0)
        self.read_tool_calls_sum += data.get("read_tool_calls", 0.0)
        self.text_scan_tool_calls_sum += data.get("text_scan_tool_calls", 0.0)
        self.generation_ms_sum += data.get("generation_ms", 0.0)
        self.total_ms_sum += data.get("total_ms", 0.0)

        if data.get("error") is None:
            self.success += 1
        else:
            self.errors += 1

    @property
    def judge_reward_mean(self) -> float:
        return self.judge_reward_sum / self.count if self.count > 0 else 0.0

    @property
    def format_reward_mean(self) -> float:
        return self.format_reward_sum / self.count if self.count > 0 else 0.0

    @property
    def general_reward_mean(self) -> float:
        return self.general_reward_sum / self.count if self.count > 0 else 0.0

    @property
    def num_turns_mean(self) -> float:
        return self.num_turns_sum / self.count if self.count > 0 else 0.0

    @property
    def total_tool_calls_mean(self) -> float:
        return self.total_tool_calls_sum / self.count if self.count > 0 else 0.0

    @property
    def search_tool_calls_mean(self) -> float:
        return self.search_tool_calls_sum / self.count if self.count > 0 else 0.0

    @property
    def read_tool_calls_mean(self) -> float:
        return self.read_tool_calls_sum / self.count if self.count > 0 else 0.0

    @property
    def text_scan_tool_calls_mean(self) -> float:
        return self.text_scan_tool_calls_sum / self.count if self.count > 0 else 0.0

    @property
    def generation_ms_mean(self) -> float:
        return self.generation_ms_sum / self.count if self.count > 0 else 0.0

    @property
    def total_ms_mean(self) -> float:
        return self.total_ms_sum / self.count if self.count > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.count if self.count > 0 else 0.0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of records."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_by_difficulty(records: List[Dict[str, Any]]) -> Dict[Any, Metrics]:
    """Analyze results grouped by difficulty."""
    grouped_metrics = defaultdict(Metrics)

    for record in records:
        info = record.get("info", {})
        difficulty = info.get("difficulty", "unknown")
        grouped_metrics[difficulty].add_sample(record)

    return grouped_metrics


def analyze_by_data_source(records: List[Dict[str, Any]]) -> Dict[str, Metrics]:
    """Analyze results grouped by data source."""
    grouped_metrics = defaultdict(Metrics)

    for record in records:
        info = record.get("info", {})
        data_source = info.get("data_source", "unknown")
        grouped_metrics[data_source].add_sample(record)

    return grouped_metrics


def print_difficulty_table(grouped_metrics: Dict[Any, Metrics], console: Console) -> None:
    """Print table grouped by difficulty, sorted by judge reward."""
    table = Table(
        title="Results by Difficulty (sorted by Judge Reward)",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Difficulty", style="cyan", justify="center")
    table.add_column("Count", justify="right")
    table.add_column("Judge Reward", justify="right", style="green")
    table.add_column("Format Reward", justify="right")
    table.add_column("Avg Reward", justify="right")
    table.add_column("Avg Turns", justify="right")
    table.add_column("Avg Tools", justify="right")
    table.add_column("Success Rate", justify="right")

    # Sort by judge reward (descending)
    sorted_items = sorted(
        grouped_metrics.items(), key=lambda x: x[1].judge_reward_mean, reverse=True
    )

    for difficulty, metrics in sorted_items:
        table.add_row(
            str(difficulty),
            str(metrics.count),
            f"{metrics.judge_reward_mean:.3f}",
            f"{metrics.format_reward_mean:.3f}",
            f"{metrics.general_reward_mean:.3f}",
            f"{metrics.num_turns_mean:.2f}",
            f"{metrics.total_tool_calls_mean:.2f}",
            f"{metrics.success_rate:.1%}",
        )

    console.print(table)


def print_data_source_table(grouped_metrics: Dict[str, Metrics], console: Console) -> None:
    """Print table grouped by data source, sorted by judge reward."""
    table = Table(
        title="Results by Data Source (sorted by Judge Reward)",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Data Source", style="magenta")
    table.add_column("Count", justify="right")
    table.add_column("Judge Reward", justify="right", style="green")
    table.add_column("Format Reward", justify="right")
    table.add_column("Avg Reward", justify="right")
    table.add_column("Avg Turns", justify="right")
    table.add_column("Avg Tools", justify="right")
    table.add_column("Success Rate", justify="right")

    # Sort by judge reward (descending)
    sorted_items = sorted(
        grouped_metrics.items(), key=lambda x: x[1].judge_reward_mean, reverse=True
    )

    for data_source, metrics in sorted_items:
        table.add_row(
            data_source,
            str(metrics.count),
            f"{metrics.judge_reward_mean:.3f}",
            f"{metrics.format_reward_mean:.3f}",
            f"{metrics.general_reward_mean:.3f}",
            f"{metrics.num_turns_mean:.2f}",
            f"{metrics.total_tool_calls_mean:.2f}",
            f"{metrics.success_rate:.1%}",
        )

    console.print(table)


def print_overall_stats(records: List[Dict[str, Any]], console: Console) -> None:
    """Print overall statistics."""
    overall = Metrics()
    for record in records:
        overall.add_sample(record)

    console.print("\n[bold]Overall Statistics:[/bold]")
    console.print(f"  Total examples: {overall.count}")
    console.print(f"  Success rate: {overall.success_rate:.1%}")
    console.print(f"  Mean judge reward: {overall.judge_reward_mean:.3f}")
    console.print(f"  Mean format reward: {overall.format_reward_mean:.3f}")
    console.print(f"  Mean general reward: {overall.general_reward_mean:.3f}")
    console.print(f"  Mean turns: {overall.num_turns_mean:.2f}")
    console.print(f"  Mean tool calls: {overall.total_tool_calls_mean:.2f}")
    console.print(f"  Mean generation time: {overall.generation_ms_mean / 1000:.1f}s")
    console.print(f"  Mean total time: {overall.total_ms_mean / 1000:.1f}s")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze RL environment evaluation results from JSONL files."
    )
    parser.add_argument("jsonl_path", type=Path, help="Path to the JSONL results file")
    args = parser.parse_args()

    console = Console()

    # Validate input
    if not args.jsonl_path.exists():
        console.print(f"[red]Error: File not found: {args.jsonl_path}[/red]")
        return 1

    console.print(f"\n[bold]Loading data from:[/bold] {args.jsonl_path}")

    # Load data
    records = load_jsonl(args.jsonl_path)
    console.print(f"[green]Loaded {len(records)} records[/green]\n")

    # Analyze by difficulty
    difficulty_metrics = analyze_by_difficulty(records)
    print_difficulty_table(difficulty_metrics, console)

    console.print()

    # Analyze by data source
    data_source_metrics = analyze_by_data_source(records)
    print_data_source_table(data_source_metrics, console)

    # Print overall stats
    print_overall_stats(records, console)

    console.print()

    return 0


if __name__ == "__main__":
    exit(main())
