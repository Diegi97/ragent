import inspect
from typing import Any, Dict, List, Union
import logging
from typing import Optional
import functools


logger = logging.getLogger(__name__)


def log_reward_error(
    title: str,
    error: Exception,
    additional_info: Optional[Dict[str, str]] = None,
    message: Optional[str] = None,
    separator_length: int = 60,
) -> None:
    """Centralized error logging function with consistent formatting."""
    separator = "=" * separator_length
    logger.error(separator)
    logger.error(f"🚨 {title} 🚨")
    logger.error(separator)
    logger.error(f"Error type: {type(error).__name__}")
    logger.error(f"Error message: {str(error)}")
    if additional_info:
        for key, value in additional_info.items():
            logger.error(f"{key}: {value}")
    if message:
        logger.error(message)
    logger.error(separator)


def safe_execution(
    error_message: str,
    default_return_value: Any = 0.0,
    additional_info_builder: Optional[callable] = None,
):
    """
    A decorator to safely execute a function (sync or async),
    logging errors and returning a default value on failure.
    """

    def handle_error(e: Exception, *args, **kwargs):
        info = {}
        if additional_info_builder:
            try:
                info = additional_info_builder(*args, **kwargs)
            except Exception as build_err:
                logger.warning(f"Failed to build additional info: {build_err}")

        log_reward_error(
            error_message,
            error=e,
            additional_info=info,
            message=f"Returning {default_return_value} to continue training...",
        )
        return default_return_value

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    return handle_error(e, *args, **kwargs)

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return handle_error(e, *args, **kwargs)

            return sync_wrapper

    return decorator


def question_from_prompt(prompt: Union[str, List[Dict[str, Any]]]) -> str:
    """Extract the question from various prompt formats."""
    if isinstance(prompt, list):
        last_msg = prompt[-1]
        if isinstance(last_msg, dict) and "content" in last_msg:
            return str(last_msg["content"])
        return ""
    return str(prompt)
