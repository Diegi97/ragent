from collections.abc import Sequence
from pathlib import Path

import anyio
from prefect import task

from data_pipelines.pipelines.deep_search_task_generation.facts import (
    EntityFactMemoryRecord,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.fact_extraction_output.diagnostics import (
    ParseDiagnostics,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.fact_extraction_output.inputs import (
    load_batch_input_metadata,
    parse_batch_output_files,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.fact_extraction_output.joining import (
    build_entity_facts_from_batch_output,
)
from data_pipelines.pipelines.deep_search_task_generation.project import phoenix_project
from data_pipelines.tracing import object_trace


@task(name="parse-fact-extraction-output", retries=0, persist_result=False)
async def parse_fact_output(
    file_paths: Sequence[Path], batch_input_path: Path
) -> tuple[dict[str, str], list[EntityFactMemoryRecord], ParseDiagnostics]:
    with object_trace(
        "parse-fact-extraction-output",
        {
            "file_paths": [str(path) for path in file_paths],
            "batch_input_path": str(batch_input_path),
        },
        {"batch.output_file_count": len(file_paths)},
        project_name=phoenix_project(),
    ) as root:
        responses, diagnostics = await anyio.to_thread.run_sync(
            parse_batch_output_files, file_paths
        )
        input_metadata = await anyio.to_thread.run_sync(
            load_batch_input_metadata, batch_input_path
        )
        entity_facts = await anyio.to_thread.run_sync(
            build_entity_facts_from_batch_output,
            responses,
            input_metadata,
            diagnostics,
        )
        root.set_output(
            {
                "response_count": len(responses),
                "entity_fact_count": len(entity_facts),
                "diagnostics": diagnostics.to_dict(),
            }
        )
        return responses, entity_facts, diagnostics
