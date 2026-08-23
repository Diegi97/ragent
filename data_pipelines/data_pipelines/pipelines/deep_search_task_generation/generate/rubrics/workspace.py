import hashlib
import shutil
import unicodedata
from pathlib import Path
from urllib.parse import quote

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.models import (
    FactWorkspace,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    validate_question_rubric_file,
)
from data_pipelines.pipelines.deep_search_task_generation.models import (
    EntityFactMemoryRecord,
)
from data_pipelines.pipelines.deep_search_task_generation.prompts import ExtractedFact


def _bucket(entity_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", entity_name.strip())
    initial = normalized[:1].upper()
    if "A" <= initial <= "Z" or initial.isdigit():
        return initial
    return "_"


def _encoded_filename(entity_name: str) -> str:
    encoded = quote(entity_name.strip(), safe=" -_.()") or "entity"
    if len(encoded) > 180:
        digest = hashlib.sha256(entity_name.encode()).hexdigest()[:12]
        encoded = f"{encoded[:160]}--{digest}"
    return f"{encoded}.md"


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _doc_ids(values: list[int] | tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _entity_fact_markdown(entity_fact: EntityFactMemoryRecord) -> str:
    lines = [
        f"# Entity: {_single_line(entity_fact.entity_name)}",
        f"Source: {_single_line(entity_fact.data_source)}",
        f"Entity docs: {_doc_ids(entity_fact.entity_doc_ids)}",
        "",
        "## Facts",
    ]
    facts_by_doc_ids: dict[tuple[int, ...], list[ExtractedFact]] = {}
    for fact in entity_fact.facts:
        facts_by_doc_ids.setdefault(tuple(fact.doc_ids), []).append(fact)
    for doc_ids, facts in facts_by_doc_ids.items():
        heading = "Document" if len(doc_ids) == 1 else "Documents"
        heading_doc_ids = ", ".join(str(value) for value in doc_ids)
        lines.extend(["", f"### {heading} {heading_doc_ids}"])
        for fact in facts:
            lines.append(f"- {_single_line(fact.statement)}")
            if fact.mentioned_entities:
                lines.append(
                    "  Mentions: "
                    + "; ".join(
                        _single_line(value) for value in fact.mentioned_entities
                    )
                )
    return "\n".join(lines) + "\n"


def _entity_index_markdown(
    entity_paths: dict[str, str],
) -> str:
    lines = [
        "# Entity index",
        "",
        "## Entities",
    ]
    lines.extend(
        f"- {_single_line(entity_name)}: {relative_path}"
        for entity_name, relative_path in entity_paths.items()
    )
    return "\n".join(lines) + "\n"


def create_fact_workspace(
    directory: Path,
    entity_facts: list[EntityFactMemoryRecord],
) -> FactWorkspace:
    directory = directory.resolve()
    facts_directory = directory / "facts"
    outputs_directory = directory / "outputs"
    audits_directory = directory / ".difficulty_checks"
    facts_directory.mkdir(parents=True, exist_ok=False)
    outputs_directory.mkdir(exist_ok=False)
    audits_directory.mkdir(exist_ok=False)

    entity_paths: dict[str, str] = {}
    used_paths: set[str] = set()
    allowed_doc_ids: set[int] = set()
    for entity_fact in entity_facts:
        bucket_directory = facts_directory / _bucket(entity_fact.entity_name)
        bucket_directory.mkdir(exist_ok=True)
        filename = _encoded_filename(entity_fact.entity_name)
        relative_path = (Path("facts") / bucket_directory.name / filename).as_posix()
        collision_key = relative_path.casefold()
        if collision_key in used_paths:
            digest = hashlib.sha256(entity_fact.entity_name.encode()).hexdigest()[:12]
            filename = f"{Path(filename).stem}--{digest}.md"
            relative_path = (
                Path("facts") / bucket_directory.name / filename
            ).as_posix()
            collision_key = relative_path.casefold()
        if collision_key in used_paths:
            raise ValueError(
                f"Could not create a unique path for {entity_fact.entity_name!r}"
            )
        used_paths.add(collision_key)
        entity_path = directory / relative_path
        entity_path.write_text(_entity_fact_markdown(entity_fact), encoding="utf-8")
        entity_paths[entity_fact.entity_name] = relative_path
        allowed_doc_ids.update(
            doc_id for fact in entity_fact.facts for doc_id in fact.doc_ids
        )

    entity_index = directory / "entity_index.md"
    entity_index.write_text(
        _entity_index_markdown(entity_paths),
        encoding="utf-8",
    )
    validator_source = Path(validate_question_rubric_file.__code__.co_filename)
    validator = directory / "validate_question_rubric.py"
    shutil.copy2(validator_source, validator)
    validator.chmod(0o444)
    scripts_directory = Path(__file__).resolve().parent
    retrieval_probe = directory / "retrieval_probe.py"
    solver = directory / "solve_question_rubric.py"
    shutil.copy2(scripts_directory / "retrieval_probe.py", retrieval_probe)
    shutil.copy2(scripts_directory / "solve_question_rubric.py", solver)
    retrieval_probe.chmod(0o444)
    solver.chmod(0o444)
    return FactWorkspace(
        directory=directory,
        facts_directory=facts_directory,
        outputs_directory=outputs_directory,
        entity_index=entity_index,
        validator=validator,
        retrieval_probe=retrieval_probe,
        solver=solver,
        audits_directory=audits_directory,
        allowed_doc_ids=frozenset(allowed_doc_ids),
    )
