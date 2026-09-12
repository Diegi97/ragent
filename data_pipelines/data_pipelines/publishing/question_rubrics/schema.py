from typing import Any

from datasets import (
    Dataset,
    Features,
    List,
    Value,
)

from ragent_core.artifacts.question_rubric import QuestionRubricDatasetRecord


def _record_features() -> Features:
    schema = QuestionRubricDatasetRecord.model_json_schema()
    return Features(_schema_feature(schema, schema.get("$defs", {})))


def _schema_feature(
    schema: dict[str, Any], definitions: dict[str, Any]
) -> Value | List | dict[str, Any]:
    if "$ref" in schema:
        return _schema_feature(
            definitions[schema["$ref"].rsplit("/", 1)[-1]], definitions
        )
    if "anyOf" in schema:
        variants = [_schema_feature(item, definitions) for item in schema["anyOf"]]
        if not variants or any(item != variants[0] for item in variants[1:]):
            raise ValueError("Rubric union needs one compatible dataset representation")
        return variants[0]
    kind = schema.get("type")
    if kind in {"string", "integer"}:
        return Value("string" if kind == "string" else "int64")
    if kind == "array":
        return List(_schema_feature(schema["items"], definitions))
    if kind == "object":
        return {
            name: _schema_feature(value, definitions)
            for name, value in schema["properties"].items()
        }
    raise ValueError(f"Unsupported rubric dataset schema type: {kind!r}")


QUESTION_RUBRIC_FEATURES = _record_features()


def align_existing_split(dataset: Dataset, features: Features) -> Dataset:
    expected_columns = list(features)
    actual_columns = set(dataset.column_names)
    if "question_type" in actual_columns:
        dataset = dataset.remove_columns("question_type")
        actual_columns = set(dataset.column_names)
    missing_columns = set(expected_columns).difference(actual_columns)
    if missing_columns == {"evolution_strategies"}:
        dataset = dataset.add_column(
            "evolution_strategies",
            [[] for _ in range(len(dataset))],
            feature=features["evolution_strategies"],
        )
        actual_columns = set(dataset.column_names)
        missing_columns = set(expected_columns).difference(actual_columns)
    extra_columns = actual_columns.difference(expected_columns)
    if missing_columns or extra_columns:
        details: list[str] = []
        if missing_columns:
            details.append("missing " + ", ".join(sorted(missing_columns)))
        if extra_columns:
            details.append("unexpected " + ", ".join(sorted(extra_columns)))
        raise ValueError(
            "Existing dataset schema is incompatible: " + "; ".join(details)
        )
    try:
        for record in dataset:
            QuestionRubricDatasetRecord.model_validate(record, strict=True)
        return dataset.select_columns(expected_columns).cast(features)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Existing dataset schema is incompatible: {exc}") from exc
