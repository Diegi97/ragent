import logging
import functools
import inspect
from typing import Callable, Any, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def retry_on_oom(min_batch_size: int = 1) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator that automatically retries a function with reduced batch_size on CUDA OOM errors.

    The decorated function must have a 'batch_size' parameter (positional or keyword).
    When a CUDA OOM error is detected, the batch_size is halved and the function is retried.
    This continues until either:
    - The function succeeds
    - The batch_size reaches min_batch_size and still fails

    Args:
        min_batch_size: Minimum batch size to try before giving up (default: 1)

    Example:
        @retry_on_oom(min_batch_size=1)
        def my_gpu_function(query: str, documents: List[str], batch_size: int = 32):
            # ... GPU processing that might OOM ...
            return results

    Raises:
        RuntimeError: If OOM persists even at min_batch_size, or if batch_size parameter not found
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Get function signature to find batch_size parameter
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            # Find batch_size in parameters
            if "batch_size" not in param_names:
                raise RuntimeError(
                    f"Function {func.__name__} must have a 'batch_size' parameter to use @retry_on_oom"
                )

            batch_size_idx = param_names.index("batch_size")

            # Determine initial batch_size value
            if batch_size_idx < len(args):
                # batch_size was passed as positional argument
                current_batch_size = args[batch_size_idx]
                args_list = list(args)
            elif "batch_size" in kwargs:
                # batch_size was passed as keyword argument
                current_batch_size = kwargs["batch_size"]
                args_list = list(args)
            else:
                # Use default value from function signature
                default = sig.parameters["batch_size"].default
                if default is inspect.Parameter.empty:
                    raise RuntimeError(
                        f"batch_size parameter in {func.__name__} has no default value and was not provided"
                    )
                current_batch_size = default
                args_list = list(args)

            # Retry loop with batch size reduction
            while current_batch_size >= min_batch_size:
                try:
                    # Update batch_size in arguments
                    if batch_size_idx < len(args):
                        # Update positional argument
                        args_list[batch_size_idx] = current_batch_size
                        result = func(*args_list, **kwargs)
                    else:
                        # Update keyword argument
                        kwargs["batch_size"] = current_batch_size
                        result = func(*args, **kwargs)

                    # Success!
                    return result

                except RuntimeError as e:
                    error_msg = str(e).lower()
                    is_oom = "out of memory" in error_msg or "cuda" in error_msg

                    if is_oom:
                        if current_batch_size > min_batch_size:
                            # Reduce batch size and retry
                            new_batch_size = max(
                                min_batch_size, current_batch_size // 2
                            )
                            logger.warning(
                                "CUDA OOM detected in %s. Reducing batch_size from %d to %d and retrying...",
                                func.__name__,
                                current_batch_size,
                                new_batch_size,
                            )
                            current_batch_size = new_batch_size

                            # Clear CUDA cache if available
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            # Already at minimum batch size, cannot reduce further
                            logger.error(
                                "CUDA OOM in %s with batch_size=%d (minimum), cannot reduce further",
                                func.__name__,
                                current_batch_size,
                            )
                            raise
                    else:
                        # Not an OOM error, re-raise immediately
                        raise

            # Should never reach here, but just in case
            raise RuntimeError(
                f"Failed to execute {func.__name__} after batch size reduction"
            )

        return wrapper

    return decorator


__all__ = ["retry_on_oom"]
