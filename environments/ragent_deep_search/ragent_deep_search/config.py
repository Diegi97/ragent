from pathlib import Path

import verifiers.v1 as vf

from ragent_core.artifacts.question_rubric import DEFAULT_RUBRIC_DATASET_ID
from ragent_deep_search.task import RagentTaskConfig
from ragent_deep_search.toolset.config import RagentToolsetConfig


class RagentConfig(vf.TasksetConfig):
    dataset_path: str | Path = DEFAULT_RUBRIC_DATASET_ID
    split: str = "test"
    num_tasks: int = 100
    task: RagentTaskConfig = RagentTaskConfig()
    tools: RagentToolsetConfig = RagentToolsetConfig()
