import re
import unicodedata
from collections import Counter
from itertools import combinations
from statistics import mean, median
from typing import Any, Mapping, Sequence

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.audit_contract import (
    SolverAudit,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.difficulty import (
    DIFFICULTY_BANDS,
    DifficultyBandName,
    difficulty_band,
    difficulty_thresholds,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation.markdown import (
    EVOLUTION_STRATEGIES,
)
from ragent_core.artifacts.question_rubric import QuestionRubricRecord
from ragent_core.judges.criteria import criterion_id


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


def _rate_entry(observations: int, passed: int) -> dict[str, int | float | None]:
    return {
        "observations": observations,
        "passed": passed,
        "pass_rate_percent": _percentage(passed, observations),
    }


def build_dataset_profile(
    accepted: Mapping[int, QuestionRubricRecord],
    solver_audits: Mapping[int, SolverAudit],
    audit_errors: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
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

    difficulty_counts: Counter[DifficultyBandName] = Counter()
    percent_passed_values: list[float] = []
    audited_item_count = 0
    audit_errors = list(audit_errors)
    criterion_observations = 0
    criteria_passed = 0
    missing_criterion_judgments = 0
    position_counts: dict[str, list[int]] = {}
    criterion_text_counts: dict[str, dict[str, Any]] = {}

    for slot, record in records:
        solver_audit = solver_audits.get(slot)
        if solver_audit is None:
            difficulty_counts[DifficultyBandName.UNKNOWN] += 1
            if not any(error.get("slot") == slot for error in audit_errors):
                audit_errors.append(
                    {
                        "slot": slot,
                        "entity": record.entity,
                        "error": "Missing validated solver audit",
                    }
                )
            continue

        audited_item_count += 1
        percent_passed_values.append(solver_audit.percent_passed)
        difficulty_counts[difficulty_band(solver_audit.percent_passed)] += 1
        judgment_by_id = {judgment.id: judgment for judgment in solver_audit.judgments}
        for index, criterion in enumerate(record.rubric, start=1):
            criterion_key = criterion_id(index)
            judgment = judgment_by_id.get(criterion_key)
            passed = judgment.passed if judgment is not None else None
            if passed is None:
                missing_criterion_judgments += 1
                continue

            criterion_observations += 1
            criteria_passed += int(passed)
            position = position_counts.setdefault(criterion_key, [0, 0])
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
        pair_jaccards.append(len(intersection) / len(union))

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
            "thresholds_percent": difficulty_thresholds(),
            "distribution": {
                name: difficulty_counts[name]
                for name in [
                    *(band.name for band in DIFFICULTY_BANDS),
                    DifficultyBandName.UNKNOWN,
                ]
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
