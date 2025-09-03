import logging
from typing import Any, Dict, List, Union

import backoff

from ragent.config import JUDGE_CLIENT, JUDGE_MODEL
from ragent.prompts.agent.search_engine import \
    RESPONSE_PARSER as AGENT_RESPONSE_PARSER
from ragent.prompts.judge.qa import PROMPT as JUDGE_PROMPT
from ragent.prompts.judge.qa import RESPONSE_PARSER as JUDGE_RESPONSE_PARSER
from ragent.utils import question_from_prompt
from ragent.utils.rewards import log_reward_error, safe_execution

logger = logging.getLogger(__name__)

# --- API Call Functions ---

"""Being very paranoid here so the API doesn't break training."""

MAX_JUDGE_CALL_TRIES = 5


@safe_execution(
    error_message=f"JUDGE API CALL FAILED AFTER ALL RETRIES! ({MAX_JUDGE_CALL_TRIES})",
    default_return_value="",
    additional_info_builder=lambda judge_prompt, *_: {
        "Judge prompt length": f"{len(judge_prompt)} characters"
    },
)
@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=MAX_JUDGE_CALL_TRIES,
    jitter=backoff.random_jitter,
    giveup=lambda e: False,  # Always retry on any exception - be very robust for training
)
def judge_api_call(judge_prompt: str) -> str:
    """Make the judge API call with backoff for any errors - bulletproof for training."""
    judge_response = JUDGE_CLIENT.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "user", "content": judge_prompt},
        ],
        seed=42,
    )
    return judge_response.choices[0].message.content


# --- Reward Functions ---


@safe_execution(
    error_message="JUDGE RESPONSE PARSING FAILED!",
    default_return_value=0.0,
    additional_info_builder=lambda content, state, *_: {"Raw content": repr(content)},
)
def parse_judge_response(content: str, state: Dict[str, Any]) -> float:
    """Parse the judge's response and convert to a reward score."""
    judge_response = JUDGE_RESPONSE_PARSER.parse_answer(content)
    state["judge_response"] = judge_response
    return 1.0 if str(judge_response).strip().upper() == "CORRECT" else 0.0


def judge_reward(
    prompt: Union[str, List[Dict[str, Any]]],
    completion: str,
    answer: str,
    state: Dict[str, Any],
    **kwargs,
) -> float:
    """Judge correctness via an external model (never let parsing errors break training)."""
    question = question_from_prompt(prompt)
    response = AGENT_RESPONSE_PARSER.parse_answer(completion)
    judge_prompt = JUDGE_PROMPT.format(
        question=question, answer=answer, response=response
    )
    cached = state.get("judge_response")
    if isinstance(cached, dict) and judge_prompt in cached:
        return cached[judge_prompt]
    content = judge_api_call(judge_prompt)
    if not content:  # If API call failed and returned empty string
        logger.warning("⚠️  Judge API call returned empty content - returning 0.0 score")
        return 0.0

    return parse_judge_response(content, state)


# --- Format Reward ---
format_reward = AGENT_RESPONSE_PARSER.get_format_reward_func()
