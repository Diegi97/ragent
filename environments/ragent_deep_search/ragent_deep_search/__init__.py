import verifiers.v1 as vf

from ragent_deep_search.config import RagentConfig
from ragent_deep_search.dataset_loader import iter_dataset_rows
from ragent_deep_search.prompt import SYSTEM_PROMPT
from ragent_deep_search.task import RagentData, RagentTask
from ragent_deep_search.toolset import RagentToolset


class RagentTaskset(vf.Taskset[RagentTask, RagentConfig]):
    @classmethod
    def toolsets(cls, config: RagentConfig) -> list[vf.Toolset]:
        return [RagentToolset.for_launch(config.tools)]

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
                system_prompt=SYSTEM_PROMPT,
                question=row["question"],
                rubric=row["rubric"],
                table_name=row["data_source"],
                doc_ids=row["doc_ids"],
                evolution_strategies=row.get("evolution_strategies") or [],
            )
            tasks.append(RagentTask(data, self.config.task))
        return tasks


__all__ = ["RagentTaskset"]
