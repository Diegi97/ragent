import re
from pathlib import Path

import verifiers.v1 as vf
from pydantic import BaseModel, Field

from ragent_core.judges.rubric import RubricJudge, RubricJudgeConfig
from ragent_deep_search.dataset_loader import DEFAULT_DATASET_ID, iter_dataset_rows
from ragent_deep_search.toolset import RagentState, RagentToolset, RagentToolsetConfig

SYSTEM_PROMPT = """\
You are a search agent. You help users by searching the corpus and reading and scanning relevant documents to answer their questions.

# Available tools

- `search`: Search the corpus for documents relevant to one or more queries.
- `read`: Read up to three documents by integer ID.
- `text_scan`: Scan the corpus for a fixed string or regular expression.

# Citations

Ground material factual claims in evidence returned by `search` or `read`. Cite the supporting document IDs inline, immediately after the relevant sentence or paragraph, using `[doc 75412]` for one document or `[docs 75412, 57846]` for multiple documents.
Cite a search result directly when its snippet clearly supports the associated claim. Use `read` when the snippet is incomplete, ambiguous, or insufficient. Cite only documents returned by `search` or successfully by `read`, and never invent document IDs, titles, or URLs.
End the answer with a `## Sources` section that lists every cited document exactly once, in order of first citation. Use `- [doc 64538] — Larry Rivers` when a title is available and `- [doc 89526]` when it is not. Use titles exactly as returned by the tools and do not list uncited documents.
"""


_CITATION_RE = re.compile(
    r"\[(?:doc|docs)\s+(?P<ids>\d+(?:\s*,\s*\d+)*)\]",
    re.IGNORECASE,
)
_DOCUMENT_RE = re.compile(
    r"<document\s+id=(?:[\"'])?(?P<id>\d+)(?:[\"'])?\s*>"
    r"(?P<content>.*?)</document\s*>",
    re.DOTALL | re.IGNORECASE,
)
_SEARCH_RESULT_RE = re.compile(
    r"<result>.*?<id>\s*(?P<id>\d+)\s*</id>.*?</result\s*>",
    re.DOTALL | re.IGNORECASE,
)
_SOURCES_HEADING_RE = re.compile(r"^## Sources\s*$", re.MULTILINE)
_MISSING_DOCUMENT_PREFIX = "Error: Document with id"

RUBRIC_REWARD_WEIGHT = 0.9
CITATION_REWARD_WEIGHT = 0.1


def _citation_ids(text: str) -> list[int]:
    return [
        int(doc_id)
        for match in _CITATION_RE.finditer(text)
        for doc_id in match.group("ids").split(",")
    ]


def _message_text(message: vf.ToolMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        part.text for part in content if isinstance(part, vf.TextContentPart)
    )


def _evidence_document_ids(trace: vf.Trace) -> set[int]:
    evidence_calls = {
        call.id: call.name
        for message in trace.assistant_messages
        for call in message.tool_calls or []
        if call.name in {"read", "search"}
    }
    evidence_ids: set[int] = set()
    for message in trace.tool_messages:
        tool_name = evidence_calls.get(message.tool_call_id)
        if tool_name is None:
            continue
        content = _message_text(message)
        if tool_name == "search":
            evidence_ids.update(
                int(match.group("id")) for match in _SEARCH_RESULT_RE.finditer(content)
            )
            continue
        for match in _DOCUMENT_RE.finditer(content):
            if not match.group("content").lstrip().startswith(_MISSING_DOCUMENT_PREFIX):
                evidence_ids.add(int(match.group("id")))
    return evidence_ids


def _citation_grounding_score(response: str, evidence_ids: set[int]) -> float:
    headings = list(_SOURCES_HEADING_RE.finditer(response))
    if not headings:
        return 0.0

    sources_heading = headings[-1]
    inline_ids = _citation_ids(response[: sources_heading.start()])
    source_ids = _citation_ids(response[sources_heading.end() :])
    expected_source_ids = list(dict.fromkeys(inline_ids))
    if not inline_ids or source_ids != expected_source_ids:
        return 0.0
    return float(set(inline_ids) <= evidence_ids)


class RagentRubricItem(BaseModel):
    criterion: str
    doc_ids: list[int | str] = Field(default_factory=list)


class RagentData(vf.TaskData):
    question: str
    rubric: list[RagentRubricItem]
    table_name: str
    doc_ids: list[int | str]
    evolution_strategies: list[str] = Field(default_factory=list)


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
    def citation_grounding(self, trace: vf.Trace) -> float:
        return _citation_grounding_score(
            trace.last_reply,
            _evidence_document_ids(trace),
        )


class RagentConfig(vf.TasksetConfig):
    dataset_path: str | Path = DEFAULT_DATASET_ID
    split: str = "test"
    num_tasks: int = 100
    task: RagentTaskConfig = RagentTaskConfig()
    tools: RagentToolsetConfig = RagentToolsetConfig()


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
