"""Utility functions"""
import logging

from typing import Iterator, TypeVar


_T = TypeVar("_T")


def parse_logging_level(level: str) -> int:
    """Return the matching logging level."""
    match level:
        case "CRITICAL":
            return logging.CRITICAL
        case "ERROR":
            return logging.ERROR
        case "WARNING":
            return logging.WARNING
        case "INFO":
            return logging.INFO
        case "DEBUG":
            return logging.DEBUG
        case _:
            raise ValueError(f"Invalid logging level {level}")


def chunks(list_: list[_T], chunk_size: int) -> Iterator[list[_T]]:
    """Yields the list in chunks of the specified size."""
    for idx in range(0, len(list_), chunk_size):
        yield list_[idx:idx + chunk_size]


def flatten_list_1d(list_: list[list[_T]]) -> list[_T]:
    """Returns the list flattened one level"""
    flat_list: list[_T] = []
    for row in list_:
        flat_list += row
    return flat_list
