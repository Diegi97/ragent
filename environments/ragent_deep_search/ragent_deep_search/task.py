import verifiers.v1 as vf
from pydantic import Field

from ragent_core.artifacts.question_rubric import (
    RubricCriterion,
    StrategyLabel,
    SupportingDocumentId,
)
from ragent_core.judges.rubric import RubricJudge
from ragent_core.judges.rubric.config import RubricJudgeConfig
from ragent_deep_search.citations import GroundedCitations
from ragent_deep_search.evidence import evidence_document_ids
from ragent_deep_search.toolset.config import RagentState

RUBRIC_REWARD_WEIGHT = 0.9


CITATION_REWARD_WEIGHT = 0.1


class RagentData(vf.TaskData):
    question: str
    rubric: list[RubricCriterion]
    table_name: str
    doc_ids: list[SupportingDocumentId]
    evolution_strategies: list[StrategyLabel] = Field(default_factory=list)


class RagentTaskConfig(vf.TaskConfig):
    # This is a directly called judge, like the BrowseComp example. It is intentionally
    # singular and is not placed in TaskConfig.judges, which expects judge plugins.
    judge: RubricJudgeConfig = RubricJudgeConfig(question_field="")


class RagentTask(vf.Task[RagentData, RagentState, RagentTaskConfig]):
    async def setup(self, trace: vf.Trace) -> None:
        trace.state.table_name = self.data.table_name

    @vf.reward(weight=RUBRIC_REWARD_WEIGHT)
    async def rubric(self, trace: vf.Trace) -> float:
        return await RubricJudge(self.config.judge).score(self.data, trace)

    @vf.reward(weight=CITATION_REWARD_WEIGHT)
    async def citation_grounding(self, trace: vf.Trace) -> float:
        citations = GroundedCitations.from_response(
            trace.last_reply, evidence_document_ids(trace)
        )
        return float(citations is not None)
