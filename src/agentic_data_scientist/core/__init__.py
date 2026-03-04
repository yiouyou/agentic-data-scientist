"""Core API and session management for Agentic Data Scientist."""

from agentic_data_scientist.core.api import DataScientist, FileInfo, Result, SessionConfig
from agentic_data_scientist.core.events import (
    CompletedEvent,
    ErrorEvent,
    FunctionCallEvent,
    FunctionResponseEvent,
    MessageEvent,
    UsageEvent,
    event_to_dict,
)
from agentic_data_scientist.core.history_store import HistoryStore, create_history_store_from_env


__all__ = [
    "DataScientist",
    "Result",
    "SessionConfig",
    "FileInfo",
    "MessageEvent",
    "FunctionCallEvent",
    "FunctionResponseEvent",
    "CompletedEvent",
    "ErrorEvent",
    "UsageEvent",
    "event_to_dict",
    "HistoryStore",
    "create_history_store_from_env",
]
