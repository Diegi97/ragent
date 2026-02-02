from __future__ import annotations

import importlib
from typing import Any

from ragent_core.pipelines.base import BasePipeline, PipelineMetadata
from ragent_core.pipelines.atomic import AtomicQAPipeline
from ragent_core.pipelines.multistep import MultiStepQAPipeline
from ragent_core.pipelines.explorer_agent import ExplorerAgentQAPipeline


DEFAULT_PIPELINE = "qa/atomic"


def _normalize_pipeline_name(pipeline_name: str) -> str:
    return pipeline_name.replace("-", "_").replace("/", ".")


def load_pipeline(pipeline_name: str | None = None) -> BasePipeline[Any]:
    """
    Dynamically import and instantiate a pipeline implementation.

    Args:
        pipeline_name: Path-like identifier (e.g. "qa/atomic"). If omitted,
            defaults to the repository's canonical QA pipeline.

    Returns:
        An initialized pipeline instance.
    """
    pipeline_name = pipeline_name or DEFAULT_PIPELINE
    module_path = _normalize_pipeline_name(pipeline_name)
    module = importlib.import_module(f"ragent_core.pipelines.{module_path}")

    candidate = getattr(module, "PIPELINE", None)
    if candidate is None:
        # Fall back to a conventional class name if no constant is exported.
        candidate = getattr(module, "Pipeline", None)

    if candidate is None:
        raise AttributeError(
            f"Pipeline module 'ragent_core.pipelines.{module_path}' does not define "
            "'PIPELINE' or 'Pipeline'"
        )

    if isinstance(candidate, type) and issubclass(candidate, BasePipeline):
        return candidate()

    if callable(candidate):
        instance = candidate()
        if not isinstance(instance, BasePipeline):
            raise TypeError(
                f"Factory in module '{module_path}' did not return a BasePipeline"
            )
        return instance

    raise TypeError(
        f"'PIPELINE' in module '{module_path}' must be a BasePipeline subclass or factory"
    )


__all__ = [
    "DEFAULT_PIPELINE",
    "BasePipeline",
    "PipelineMetadata",
    "load_pipeline",
    "AtomicQAPipeline",
    "MultiStepQAPipeline",
    "ExplorerAgentQAPipeline",
]
