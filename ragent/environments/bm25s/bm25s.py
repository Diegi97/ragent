import verifiers as vf
from datasets import load_dataset
from openai import OpenAI

from ragent.data.pipelines import safe_ds_name
from ragent.data.sources.bm25s import BM25Client
from ragent.reward_funcs.llm_judge_prompt import LLM_JUDGE_PROMPT, JUDGE_RESPONSE_PARSER
from ragent.data.prompts.search_engine import SEARCH_ENGINE_RESPONSE_PARSER, SEARCH_ENGINE_PROMPT


def load_environment(hf_dataset: str = "nampdn-ai/devdocs.io", llm_name_judge: str = "google/gemini-2.5-flash", **kwargs) -> vf.Environment:

    bm25_client = BM25Client(hf_dataset)
    qas_dataset = load_dataset(f"data/{safe_ds_name(hf_dataset)}/", data_files="qas.json", split="train")

    # openrouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
    )

    rubric = vf.JudgeRubric(
        parser=SEARCH_ENGINE_RESPONSE_PARSER,
        parallelize_scoring=True,
        judge_prompt=LLM_JUDGE_PROMPT,
        judge_client=client,
        judge_model=llm_name_judge,
    )

    def grade_reward(prompt, completion, answer, state, **kwargs):
        judge_response = rubric.judge(prompt, completion, answer, state, **kwargs)
        judge_response = JUDGE_RESPONSE_PARSER.parse_answer(judge_response)
        return 1.0 if judge_response == "CORRECT" else 0.0

    rubric.add_reward_func(grade_reward, 0.8)
    rubric.add_reward_func(SEARCH_ENGINE_RESPONSE_PARSER.get_format_reward_func(), 0.2)

    return vf.ToolEnv(
        dataset=qas_dataset,
        rubric=rubric,
        system_prompt=SEARCH_ENGINE_PROMPT,
        tools=[bm25_client.search_tool, bm25_client.read_tool], # python functions with type hints + docstrings
        max_turns=10
    )
