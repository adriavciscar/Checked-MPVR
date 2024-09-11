"""Agents and utilities for agents"""
# pylint: disable=unnecessary-ellipsis
from typing import Optional, Protocol, Type, Any

from agents.numbered_cmpvr import NumberedCMPVRAgent

from .checked_mpvr import CheckedMPVRAgent
from .mpvr import MPVRAgent


__all__ = ["CURRENT_LLM_MODELS", "Agent", "CURRENT_AGENTS", "get_agent"]

CURRENT_LLM_MODELS = {
    "gemma-7b-it",
    "gemma2-9b-it",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
}


class Agent(Protocol):
    """Agent Base class"""

    def __init__(self, query_model_name: str, prompt_model_name: str,
                 query_model_args: Optional[dict[str, Any]] = None,
                 prompt_model_args: Optional[dict[str, Any]] = None) -> None:
        ...

    def get_queries(self, *,
                    dataset_name: str, dataset_description: str,
                    amount: int) -> list[str]:
        """Obtain the queries for a dataset using a chain."""
        ...

    def get_prompts(self,
                    queries: list[str],
                    categories: list[str]) -> list[list[str]]:
        """Get the prompts for the queries and categories."""
        ...


CURRENT_AGENTS: dict[str, Type[Agent]] = {
    "mpvr": MPVRAgent,
    "checked_mpvr": CheckedMPVRAgent,
    "numbered_cmpvr": NumberedCMPVRAgent
}


def get_agent(agent_name: str) -> Type[Agent]:
    """Gets the Agent class of the agent name."""
    return CURRENT_AGENTS[agent_name]
