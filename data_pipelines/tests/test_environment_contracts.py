import json
import threading
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from datasets import Dataset

from ragent_core.retrievers.agent_retriever import AgentRetriever
from ragent_core.retrievers.document import DOCUMENT_ID_KEY, Document, RetrievalResult
from ragent_core.retrievers.settings import TURBOPUFFER_API_KEY_ENV
from ragent_core.retrievers.tool_protocol import ToolName
from ragent_deep_search import RagentTaskset, dataset_loader
from ragent_deep_search import task as task_module
from ragent_deep_search.citations import GroundedCitations
from ragent_deep_search.config import RagentConfig
from ragent_deep_search.dataset_loader import iter_dataset_rows
from ragent_deep_search.evidence import evidence_document_ids
from ragent_deep_search.task import (
    CITATION_REWARD_WEIGHT,
    RUBRIC_REWARD_WEIGHT,
    RagentData,
    RagentTask,
)
from ragent_deep_search.toolset import RagentToolset
from ragent_deep_search.toolset.config import RagentState, RagentToolsetConfig


def rubric_row(source="row-source"):
    return {
        "entity": "Entity",
        "question": "Question",
        "rubric": [{"criterion": "Requirement", "doc_ids": [0]}],
        "doc_ids": [0],
        "data_source": source,
    }


def test_local_metadata_is_authoritative_and_taskset_remains_discoverable(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(rubric_row()))
    (tmp_path / dataset_loader.METADATA_FILENAME).write_text(
        json.dumps(
            {
                "prepare_config": {"data_source": " metadata-source ", "seed": 42},
                "diagnostics": {"count": 1},
            }
        )
    )
    taskset = RagentTaskset(RagentConfig(dataset_path=path, num_tasks=1))
    tasks = taskset.load()
    assert len(tasks) == 1
    assert tasks[0].data.table_name == "metadata-source"
    assert tasks[0].data.doc_ids == [0]
    assert tasks[0].data.evolution_strategies == []


def test_hub_rows_use_same_rubric_validation(monkeypatch):

    row = rubric_row()
    row["doc_ids"] = [1]
    monkeypatch.setattr(
        dataset_loader, "load_dataset", lambda *args, **kwargs: Dataset.from_list([row])
    )
    with pytest.raises(ValueError, match="union"):
        list(iter_dataset_rows("example/dataset", "test"))


@pytest.mark.parametrize(
    "answer",
    [
        "Fact [doc 0]",
        "Fact [doc 0]\n## Sources\n[doc 1]",
        "Fact [docs 0,1]\n## Sources\n[doc 1]\n[doc 0]",
        "Fact [doc 0]\n## Sources\n[doc 0]\n[doc 0]",
    ],
)
def test_citation_contract_rejects_missing_or_mismatched_sources(answer):
    assert GroundedCitations.from_response(answer, {0, 1}) is None


def test_search_and_successful_read_xml_supply_citation_evidence():
    class Backend:
        def retrieve(self, *args, **kwargs):
            return [
                RetrievalResult(id=3, content="snippet", metadata={DOCUMENT_ID_KEY: 0})
            ]

        def get_document(self, doc_id, table_name):
            return Document(id=1, content="Full document") if doc_id == 1 else None

    tools = AgentRetriever(Backend())
    trace = SimpleNamespace(
        assistant_messages=[
            SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(id="search", name=ToolName.SEARCH),
                    SimpleNamespace(id="read", name=ToolName.READ),
                ]
            )
        ],
        tool_messages=[
            SimpleNamespace(
                tool_call_id="search", content=tools.search_tool(["query"], "source")
            ),
            SimpleNamespace(
                tool_call_id="read", content=tools.read_tool([1, 2], "source")
            ),
        ],
    )
    evidence = evidence_document_ids(trace)
    assert evidence == {0, 1}
    assert GroundedCitations.from_response(
        "Facts [docs 0,1]. Again [doc 0].\n## Sources\n[doc 0]\n[doc 1]", evidence
    ).document_ids == (0, 1)


@pytest.mark.asyncio
async def test_tool_setup_offloads_loading_and_direct_and_decorated_reads_share_validation(
    monkeypatch,
):
    owner_thread = threading.get_ident()
    loader_threads = []
    calls = []

    class Retriever:
        def read_tool(self, ids, table_name):
            calls.append((ids, table_name))
            return "read result"

    def load(**kwargs):
        loader_threads.append(threading.get_ident())
        return Retriever()

    monkeypatch.setenv(TURBOPUFFER_API_KEY_ENV, "fake-test-key")
    monkeypatch.setattr(AgentRetriever, "from_turbopuffer_index", load)
    toolset = RagentToolset(RagentToolsetConfig())
    await toolset.setup()
    toolset.state.table_name = "source"
    assert loader_threads and loader_threads[0] != owner_thread
    assert "only integers" in await toolset.read([True], table_name="source")
    assert "only integers" in await toolset.read_tool(["0"])
    assert calls == []
    assert await toolset.read_tool([0]) == "read result"
    assert calls == [([0], "source")]


def test_tool_api_key_environment_overrides_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{TURBOPUFFER_API_KEY_ENV}=file-key\n")
    config = RagentToolsetConfig(env_file=env_file)
    monkeypatch.delenv(TURBOPUFFER_API_KEY_ENV, raising=False)
    assert config.turbopuffer_api_key() == "file-key"
    monkeypatch.setenv(TURBOPUFFER_API_KEY_ENV, "environment-key")
    assert config.turbopuffer_api_key() == "environment-key"


@pytest.mark.parametrize("prepare_config", [{}, {"data_source": " "}, []])
def test_local_dataset_metadata_rejects_invalid_source_before_rows(
    tmp_path, prepare_config
):
    path = tmp_path / "questions.jsonl"
    path.write_text("not a valid row")
    (tmp_path / dataset_loader.METADATA_FILENAME).write_text(
        json.dumps({"prepare_config": prepare_config})
    )
    with pytest.raises(ValueError, match="local task metadata"):
        list(iter_dataset_rows(path, "test"))


@pytest.mark.asyncio
@pytest.mark.parametrize("has_citation", [True, False])
async def test_task_framework_dispatches_rewards_and_preserves_weights(
    monkeypatch, has_citation
):
    data = RagentData(
        question="Question",
        rubric=[{"criterion": "Requirement", "doc_ids": [0]}],
        table_name="source",
        doc_ids=[0],
    )
    task = RagentTask(data)
    answer = "Answer [doc 0]\n## Sources\n[doc 0]" if has_citation else "Answer"
    trace = vf.Trace(
        task=vf.TraceTask(type="RagentTask", data=data),
        agent=vf.AgentInfo(config=vf.AgentConfig(), trainable=False),
        state=RagentState(),
        nodes=[
            vf.MessageNode(message=vf.AssistantMessage(content=answer), sampled=True)
        ],
    )
    judged = []

    class Judge:
        def __init__(self, config):
            assert config == task.config.judge

        async def score(self, actual_data, actual_trace):
            judged.append((actual_data, actual_trace))
            return 0.5

    monkeypatch.setattr(task_module, "RubricJudge", Judge)
    monkeypatch.setattr(task_module, "evidence_document_ids", lambda actual_trace: {0})
    await task.setup(trace)
    await task.score(trace)
    assert trace.state.table_name == data.table_name
    assert judged == [(data, trace)]
    assert set(trace.rewards) == {"rubric", "citation_grounding"}
    assert trace.rewards["rubric"].score == 0.5
    assert trace.rewards["rubric"].weight == RUBRIC_REWARD_WEIGHT
    assert trace.rewards["citation_grounding"].score == float(has_citation)
    assert trace.rewards["citation_grounding"].weight == CITATION_REWARD_WEIGHT
    assert trace.reward == pytest.approx(
        0.5 * RUBRIC_REWARD_WEIGHT + has_citation * CITATION_REWARD_WEIGHT
    )
