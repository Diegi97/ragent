from typing import Any, Dict, List, Union


def question_from_prompt(prompt: Union[str, List[Dict[str, Any]]]) -> str:
    """Extract the question from various prompt formats."""
    if isinstance(prompt, list):
        last_msg = prompt[-1]
        if isinstance(last_msg, dict) and "content" in last_msg:
            return str(last_msg["content"])
        return ""
    return str(prompt)
