import json
import random
from pathlib import Path
from typing import Sequence

from data_pipelines.pipelines.deep_search_task_generation.facts import (
    EntityFactMemoryRecord,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.config import (
    RubricGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.models import (
    QuestionRubricAssignment,
)


def order_entity_facts(
    entity_facts: Sequence[EntityFactMemoryRecord], entities_path: Path
) -> list[EntityFactMemoryRecord]:
    records_by_name = {record.entity_name: record for record in entity_facts}
    ordered: list[EntityFactMemoryRecord] = []
    seen: set[str] = set()
    with entities_path.open(encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed entity record at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Entity record at line {line_number} is not a JSON object."
                )
            entity_name = str(value.get("name") or "").strip()
            record = records_by_name.get(entity_name)
            if record is not None and entity_name not in seen:
                ordered.append(record)
                seen.add(entity_name)
    ordered.extend(record for record in entity_facts if record.entity_name not in seen)
    return ordered


def build_question_rubric_assignments(
    entity_facts: Sequence[EntityFactMemoryRecord],
    config: RubricGenerationConfig,
) -> list[QuestionRubricAssignment]:
    usable = [record for record in entity_facts if record.facts]
    if not usable:
        return []
    if config.random_entities:
        if config.num_question_rubrics > len(usable):
            raise ValueError(
                "random entity selection without replacement requires at least "
                f"{config.num_question_rubrics} usable entities; found {len(usable)}"
            )
        selected = random.Random(config.seed).sample(
            usable, k=config.num_question_rubrics
        )
        return [
            QuestionRubricAssignment(slot=slot, entity_fact=entity_fact)
            for slot, entity_fact in enumerate(selected)
        ]
    return [
        QuestionRubricAssignment(slot=slot, entity_fact=usable[slot % len(usable)])
        for slot in range(config.num_question_rubrics)
    ]
