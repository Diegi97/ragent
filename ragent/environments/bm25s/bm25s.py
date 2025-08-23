import verifiers as vf
from datasets import load_dataset

from ragent.data.pipelines import safe_ds_name
from ragent.data.sources.bm25s import BM25Client
from ragent.prompts.agent.search_engine import PROMPT as AGENT_PROMPT
from ragent.rewards import format_reward, judge_reward

JUDGE_REWARD_FUNC_WEIGHT = 0.8
GET_FORMAT_REWARD_FUNC_WEIGHT = 0.2


def load_environment(
    hf_dataset: str = "nampdn-ai/devdocs.io", **kwargs
) -> vf.Environment:
    bm25_client = BM25Client(hf_dataset)
    qas_dataset = load_dataset(
        f"data/{safe_ds_name(hf_dataset)}/", data_files="qas.json", split="train"
    )

    rubric = vf.Rubric(
        funcs=[judge_reward, format_reward],
        weights=[JUDGE_REWARD_FUNC_WEIGHT, GET_FORMAT_REWARD_FUNC_WEIGHT],
    )

    return vf.ToolEnv(
        dataset=qas_dataset,
        rubric=rubric,
        system_prompt=AGENT_PROMPT,
        tools=[bm25_client.search_tool, bm25_client.read_tool],
        max_turns=10,
    )
