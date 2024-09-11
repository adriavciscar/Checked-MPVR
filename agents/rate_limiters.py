"""Module for the rate limiter creations"""
from typing import Optional
from langchain_core.rate_limiters import InMemoryRateLimiter


__all__ = ["get_rate_limiter"]

_MODEL_ARGS: dict[str, dict[str, float]] = {
    "gemma-7b-it": {
        "requests_per_second": 0.45,
        "check_every_n_seconds": 1.0
    },
    "gemma2-9b-it": {
        "requests_per_second": 0.45,
        "check_every_n_seconds": 1.0
    },
    "llama-3.1-70b-versatile": {
        "requests_per_second": 0.45,
        "check_every_n_seconds": 1.0
    },
    "llama-3.1-8b-instant": {
        "requests_per_second": 0.45,
        "check_every_n_seconds": 1.0
    },
    "llama-guard-3-8b": {
        "requests_per_second": 0.45,
        "check_every_n_seconds": 1.0
    },
    "llama3-70b-8192": {
        "requests_per_second": 0.45,
        "check_every_n_seconds": 1.0
    },
    "llama3-8b-8192": {
        "requests_per_second": 0.45,
        "check_every_n_seconds": 1.0
    },
    "mixtral-8x7b-32768": {
        "requests_per_second": 0.45,
        "check_every_n_seconds": 1.0
    },
}


def get_rate_limiter(
    model_name: str, *,
    requests_per_second: Optional[float] = None,
    check_every_n_seconds: Optional[float] = None,
    max_bucket_size: Optional[int] = None
) -> InMemoryRateLimiter:
    """Gets the RateLimiter suitable for a model."""
    try:
        rate_kwargs = _MODEL_ARGS[model_name]
    except KeyError as error:
        raise ValueError("Unsupported model name") from error
    if requests_per_second is not None:
        rate_kwargs["requests_per_second"] = requests_per_second
    if check_every_n_seconds is not None:
        rate_kwargs["check_every_n_seconds"] = check_every_n_seconds
    if max_bucket_size is not None:
        rate_kwargs["max_bucket_size"] = max_bucket_size
    return InMemoryRateLimiter(**rate_kwargs)
