import verifiers as vf

from ragent.config import JUDGE_CLIENT, JUDGE_MODEL
from ragent.prompts.agent.search_engine import \
    RESPONSE_PARSER as AGENT_RESPONSE_PARSER
from ragent.prompts.judge.qa import PROMPT as JUDGE_PROMPT
from ragent.prompts.judge.qa import RESPONSE_PARSER as JUDGE_RESPONSE_PARSER


# --- Judge Reward ---
def judge_reward(prompt, completion, answer, state, **kwargs):
    r = vf.JudgeRubric(
        judge_model=JUDGE_MODEL,
        judge_prompt=JUDGE_PROMPT,
        judge_client=JUDGE_CLIENT,
        parser=AGENT_RESPONSE_PARSER,
        parallelize_scoring=True,
    )
    judge_response = r.judge(prompt, completion, answer, state, **kwargs)
    judge_response = JUDGE_RESPONSE_PARSER.parse_answer(judge_response)
    return 1.0 if judge_response == "CORRECT" else 0.0


# --- Format Reward ---
format_reward = AGENT_RESPONSE_PARSER.get_format_reward_func()
