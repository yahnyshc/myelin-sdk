"""Shared mutable state between Myelin LangChain tools and handler."""

from __future__ import annotations

from dataclasses import dataclass

from ...types import WorkflowInfo


@dataclass
class _MyelinToolState:
    """Shared state that tools write and the callback handler reads.

    Tools update ``session_id`` and ``workflow`` on recall; the handler
    reads ``session_id`` to tag captured tool calls. ``active`` tracks
    whether a session is in flight (used for auto-finish on exit).
    """

    session_id: str | None = None
    matched: bool = False
    workflow: WorkflowInfo | None = None
    active: bool = False
