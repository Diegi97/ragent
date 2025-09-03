import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

def log_reward_error(
    title: str,
    error: Exception,
    additional_info: Optional[Dict[str, str]] = None,
    message: Optional[str] = None,
    separator_length: int = 60
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

import functools
# ... other imports from your original code

def safe_execution(
    error_message: str,
    default_return_value: Any = 0.0,
    additional_info_builder: callable = None
):
    """
    A decorator to safely execute a function, logging errors and returning a
    default value on failure.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Build additional info if a builder function is provided
                info = {}
                if additional_info_builder:
                    info = additional_info_builder(*args, **kwargs)

                log_reward_error(
                    error_message,
                    error=e,
                    additional_info=info,
                    message=f"Returning {default_return_value} to continue training..."
                )
                return default_return_value
        return wrapper
    return decorator