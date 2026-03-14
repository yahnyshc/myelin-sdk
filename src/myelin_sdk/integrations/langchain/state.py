"""Shared mutable state between Myelin LangChain tools and handler."""

from __future__ import annotations

from dataclasses import dataclass

from ...types import SearchResult


@dataclass
class _MyelinToolState:
    """Shared state that tools write and the callback handler reads.

    Tools update ``session_id`` on start; the handler reads ``session_id``
    to tag captured tool calls. ``active`` tracks whether a session is in
    flight (used for auto-finish on exit). ``last_search`` holds the most
    recent search result for reference.
    """

    session_id: str | None = None
    matched_workflow_id: str | None = None
    last_search: SearchResult | None = None
    active: bool = False
