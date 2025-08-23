import logging

import backoff

from ragent.config import JUDGE_CLIENT, JUDGE_MODEL
from ragent.prompts.agent.search_engine import RESPONSE_PARSER as AGENT_RESPONSE_PARSER
from ragent.prompts.judge.qa import PROMPT as JUDGE_PROMPT
from ragent.prompts.judge.qa import RESPONSE_PARSER as JUDGE_RESPONSE_PARSER

# Set up logger for judge failures
logger = logging.getLogger(__name__) 


"""Being very paranoid here so the API doesn't break training."""


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    jitter=backoff.random_jitter,
    # Always retry on any exception - be very robust for training
    giveup=lambda e: False
)
def _make_judge_api_call(judge_prompt):
    """Make the judge API call with backoff for any errors - bulletproof for training."""
    judge_response = JUDGE_CLIENT.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "user", "content": judge_prompt},
        ],
        seed=42,
    )
    return judge_response.choices[0].message.content


def _safe_judge_api_call(judge_prompt):
    """Wrapper that ensures we never raise exceptions during training."""
    try:
        return _make_judge_api_call(judge_prompt)
    except Exception as e:
        # If all retries failed, log prominently and return empty string to keep training going
        logger.error("=" * 80)
        logger.error("🚨 JUDGE API CALL FAILED AFTER ALL RETRIES! 🚨")
        logger.error("=" * 80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Judge prompt length: {len(judge_prompt)} characters")
        logger.error("Returning 0.0 score to continue training...")
        logger.error("=" * 80)
        return ""


# --- Judge Reward ---
def judge_reward(prompt, completion, answer, state, **kwargs):
    """Judge correctness via OpenRouter using the OpenAI client."""
    if isinstance(prompt, list):
        last_msg = prompt[-1]
        if isinstance(last_msg, dict) and "content" in last_msg:
            question = str(last_msg["content"])
        else:
            question = ""
    else:
        question = str(prompt)
    response = AGENT_RESPONSE_PARSER.parse_answer(completion)
    judge_prompt = JUDGE_PROMPT.format(
        question=question, answer=answer, response=response
    )
    cached = state.get("judge_response")
    if isinstance(cached, dict) and judge_prompt in cached:
        return cached[judge_prompt]
    content = _safe_judge_api_call(judge_prompt)
    if not content:  # If API call failed and returned empty string
        logger.warning("⚠️  Judge API call returned empty content - returning 0.0 score")
        return 0.0
    
    # Safely parse the response - never let parsing errors break training
    try:
        judge_response = JUDGE_RESPONSE_PARSER.parse_answer(content)
        state["judge_response"] = judge_response
        return 1.0 if str(judge_response).strip().upper() == "CORRECT" else 0.0
    except Exception as e:
        # If parsing fails, log prominently and return 0.0 to keep training going
        logger.error("=" * 60)
        logger.error("🚨 JUDGE RESPONSE PARSING FAILED! 🚨")
        logger.error("=" * 60)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Raw content: {repr(content)}")
        logger.error("Returning 0.0 score to continue training...")
        logger.error("=" * 60)
        return 0.0


# --- Format Reward ---
format_reward = AGENT_RESPONSE_PARSER.get_format_reward_func()
