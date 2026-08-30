import json
import re
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.models import (
    QuestionRubricAssignment,
    QuestionRubricRecord,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    EVOLUTION_STRATEGIES,
)

DIFFICULTY_BANDS = ("easy", "middle", "hard", "very_hard", "unknown")


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    return " ".join(re.sub(r"[^\w]+", " ", value).split())


CANONICAL_STRATEGIES = {
    _normalized_text(strategy): strategy for strategy in EVOLUTION_STRATEGIES
}


def normalize_evolution_strategy(value: str) -> tuple[str, bool]:
    normalized = _normalized_text(value)
    canonical = CANONICAL_STRATEGIES.get(normalized)
    if canonical is not None:
        return canonical, True
    return normalized or "unknown", False


def _percentage(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(100 * numerator / denominator, 2)


def _numeric_summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": min(values),
        "mean": round(mean(values), 2),
        "median": round(median(values), 2),
        "max": max(values),
    }


def _difficulty_band(percent_passed: float) -> str:
    if not 0 <= percent_passed <= 100:
        return "unknown"
    if percent_passed >= 85:
        return "easy"
    if percent_passed >= 60:
        return "middle"
    if percent_passed >= 40:
        return "hard"
    return "very_hard"


def _load_solver_audit(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("solver audit must be a JSON object")
    return value


def _rate_entry(observations: int, passed: int) -> dict[str, int | float | None]:
    return {
        "observations": observations,
        "passed": passed,
        "pass_rate_percent": _percentage(passed, observations),
    }


def build_dataset_profile(
    assignments: Sequence[QuestionRubricAssignment],
    accepted: Mapping[int, QuestionRubricRecord],
    audits_directory: Path,
) -> dict[str, Any]:
    assignment_by_slot = {assignment.slot: assignment for assignment in assignments}
    records = [(slot, accepted[slot]) for slot in sorted(accepted)]
    item_count = len(records)

    strategy_counts: Counter[str] = Counter()
    unknown_strategy_counts: Counter[str] = Counter()
    items_without_evolution = 0
    criterion_counts: list[int] = []
    document_sets: list[set[int]] = []
    document_item_counts: Counter[int] = Counter()

    for _, record in records:
        normalized_strategies: dict[str, bool] = {}
        for strategy in record.evolution_strategies:
            label, recognized = normalize_evolution_strategy(strategy)
            normalized_strategies[label] = recognized
        if not normalized_strategies:
            items_without_evolution += 1
        for label, recognized in normalized_strategies.items():
            strategy_counts[label] += 1
            if not recognized:
                unknown_strategy_counts[label] += 1

        criterion_counts.append(len(record.rubric))
        item_doc_ids = set(record.doc_ids)
        document_sets.append(item_doc_ids)
        document_item_counts.update(item_doc_ids)

    difficulty_counts: Counter[str] = Counter()
    percent_passed_values: list[float] = []
    audited_item_count = 0
    audit_errors: list[dict[str, Any]] = []
    criterion_observations = 0
    criteria_passed = 0
    missing_criterion_judgments = 0
    position_counts: dict[str, list[int]] = {}
    criterion_text_counts: dict[str, dict[str, Any]] = {}

    for slot, record in records:
        assignment = assignment_by_slot.get(slot)
        filename = (
            assignment.filename
            if assignment is not None
            else f"question_rubric_{slot:06d}.md"
        )
        audit_path = audits_directory / f"{filename}.solver.json"
        try:
            solver_audit = _load_solver_audit(audit_path)
        except (OSError, ValueError) as exc:
            difficulty_counts["unknown"] += 1
            audit_errors.append(
                {
                    "slot": slot,
                    "entity": record.entity,
                    "path": str(audit_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        audited_item_count += 1
        percent_passed = solver_audit.get("percent_passed")
        if (
            isinstance(percent_passed, int | float)
            and not isinstance(percent_passed, bool)
            and 0 <= percent_passed <= 100
        ):
            numeric_percent = float(percent_passed)
            percent_passed_values.append(numeric_percent)
            difficulty_counts[_difficulty_band(numeric_percent)] += 1
        else:
            difficulty_counts["unknown"] += 1

        judgments = solver_audit.get("judgments")
        judgment_by_id: dict[str, dict[str, Any]] = {}
        if isinstance(judgments, list):
            for judgment in judgments:
                if not isinstance(judgment, dict):
                    continue
                criterion_id = judgment.get("id")
                if isinstance(criterion_id, str):
                    judgment_by_id[criterion_id] = judgment
        for index, criterion in enumerate(record.rubric, start=1):
            criterion_id = f"C-{index:03d}"
            judgment = judgment_by_id.get(criterion_id)
            passed = judgment.get("passed") if isinstance(judgment, dict) else None
            if not isinstance(passed, bool):
                missing_criterion_judgments += 1
                continue

            criterion_observations += 1
            criteria_passed += int(passed)
            position = position_counts.setdefault(criterion_id, [0, 0])
            position[0] += 1
            position[1] += int(passed)

            normalized_criterion = _normalized_text(criterion.criterion)
            criterion_group = criterion_text_counts.setdefault(
                normalized_criterion,
                {
                    "criterion": criterion.criterion,
                    "observations": 0,
                    "passed": 0,
                },
            )
            criterion_group["observations"] += 1
            criterion_group["passed"] += int(passed)

    criterion_count_distribution = Counter(criterion_counts)
    item_doc_reference_count = sum(len(doc_ids) for doc_ids in document_sets)
    pair_count = 0
    overlapping_pair_count = 0
    pair_jaccards: list[float] = []
    for first, second in combinations(document_sets, 2):
        pair_count += 1
        intersection = first.intersection(second)
        union = first.union(second)
        overlapping_pair_count += int(bool(intersection))
        pair_jaccards.append(len(intersection) / len(union) if union else 0.0)

    per_criterion_pass_rates = []
    for group in criterion_text_counts.values():
        entry = {
            **group,
            "pass_rate_percent": _percentage(group["passed"], group["observations"]),
        }
        per_criterion_pass_rates.append(entry)
    per_criterion_pass_rates.sort(
        key=lambda entry: (
            -entry["observations"],
            entry["pass_rate_percent"],
            entry["criterion"],
        )
    )

    return {
        "accepted_item_count": item_count,
        "evolution_strategies": {
            "distribution": dict(strategy_counts.most_common()),
            "unrecognized_distribution": dict(unknown_strategy_counts.most_common()),
            "items_without_evolution": items_without_evolution,
            "strategy_application_count": sum(strategy_counts.values()),
        },
        "difficulty": {
            "thresholds_percent": {
                "easy": "85-100",
                "middle": "60-<85",
                "hard": "40-<60",
                "very_hard": "0-<40",
            },
            "distribution": {
                band: difficulty_counts[band] for band in DIFFICULTY_BANDS
            },
            "percent_passed": _numeric_summary(percent_passed_values),
        },
        "criteria": {
            "total": sum(criterion_counts),
            "per_item": _numeric_summary(criterion_counts),
            "count_distribution": {
                str(count): frequency
                for count, frequency in sorted(criterion_count_distribution.items())
            },
        },
        "document_coverage": {
            "unique_document_count": len(document_item_counts),
            "item_document_reference_count": item_doc_reference_count,
            "unique_to_reference_ratio_percent": _percentage(
                len(document_item_counts), item_doc_reference_count
            ),
            "documents_reused_across_items": sum(
                count > 1 for count in document_item_counts.values()
            ),
            "max_items_per_document": max(document_item_counts.values(), default=0),
            "max_document_item_share_percent": _percentage(
                max(document_item_counts.values(), default=0), item_count
            ),
            "top_reused_documents": [
                {
                    "doc_id": doc_id,
                    "item_count": count,
                    "item_share_percent": _percentage(count, item_count),
                }
                for doc_id, count in document_item_counts.most_common(20)
            ],
            "item_pair_count": pair_count,
            "item_pairs_with_overlap": overlapping_pair_count,
            "pair_overlap_rate_percent": _percentage(
                overlapping_pair_count, pair_count
            ),
            "mean_pair_jaccard_percent": (
                round(100 * mean(pair_jaccards), 2) if pair_jaccards else None
            ),
        },
        "solver_audits": {
            "rollouts_per_item": 1,
            "audited_item_count": audited_item_count,
            "audit_error_count": len(audit_errors),
            "audit_errors": audit_errors,
            "criterion_observation_count": criterion_observations,
            "criteria_passed": criteria_passed,
            "overall_criterion_pass_rate_percent": _percentage(
                criteria_passed, criterion_observations
            ),
            "missing_criterion_judgment_count": missing_criterion_judgments,
            "pass_rates_by_position": {
                criterion_id: _rate_entry(*counts)
                for criterion_id, counts in sorted(position_counts.items())
            },
            "pass_rates_by_normalized_criterion": per_criterion_pass_rates,
        },
    }
