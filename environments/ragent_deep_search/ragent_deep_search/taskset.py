from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field

from ragent_core.judges.rubric import RubricJudge, RubricJudgeConfig
from ragent_deep_search.dataset_loader import DEFAULT_DATASET_ID, iter_dataset_rows
from ragent_deep_search.toolset import RagentState, RagentToolset, RagentToolsetConfig


class RagentRubricItem(vf.StrictBaseModel):
    criterion: str
    doc_ids: list[int] = Field(default_factory=list)


class RagentData(vf.TaskData):
    question: str
    rubric: list[RagentRubricItem]
    table_name: str
    doc_ids: list[int]
    complexity: str


class RagentTaskConfig(vf.TaskConfig):
    # This is a directly called judge, like the BrowseComp example. It is intentionally
    # singular and is not placed in TaskConfig.judges, which expects judge plugins.
    judge: RubricJudgeConfig = RubricJudgeConfig(question_field="")


class RagentTask(vf.Task[RagentData, RagentState, RagentTaskConfig]):
    async def setup(self, trace: vf.Trace) -> None:
        trace.state.table_name = self.data.table_name

    @vf.reward
    async def rubric(self, trace: vf.Trace) -> float:
        return await RubricJudge(self.config.judge).score(self.data, trace)


class RagentConfig(vf.TasksetConfig):
    dataset_path: str | Path = DEFAULT_DATASET_ID
    split: str = "test"
    num_tasks: int = 100
    task: RagentTaskConfig = RagentTaskConfig()
    tools: RagentToolsetConfig = RagentToolsetConfig()


class RagentTaskset(vf.Taskset[RagentTask, RagentConfig]):
    tools = (RagentToolset,)

    def load(self) -> list[RagentTask]:
        tasks: list[RagentTask] = []
        for idx, row in iter_dataset_rows(
            self.config.dataset_path,
            self.config.split,
        ):
            if len(tasks) >= self.config.num_tasks:
                break

            data = RagentData(
                idx=idx,
                prompt=row["question"],
                question=row["question"],
                rubric=row["rubric"],
                table_name=row["data_source"],
                doc_ids=row["doc_ids"],
                complexity=row["question_type"],
            )
            tasks.append(RagentTask(data, self.config.task))
        return tasks
