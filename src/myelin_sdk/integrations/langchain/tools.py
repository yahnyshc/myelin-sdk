"""LangChain tools for autonomous Myelin search/start/finish."""

from __future__ import annotations

import json
import logging
from typing import Any, Type

try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field, PrivateAttr
except ImportError:
    raise ImportError(
        "Install langchain support: pip install myelin-sdk[langchain]"
    )

from ...client import MyelinClient
from .state import _MyelinToolState

logger = logging.getLogger(__name__)


# -- Input schemas -----------------------------------------------------------


class SearchInput(BaseModel):
    """Input for memory_search."""

    task_description: str = Field(
        description=(
            "Natural language description of the task. "
            "Keep procedural detail (steps, output format, counts); "
            "omit only variable values (file paths, names, IDs) "
            "that change between uses."
        )
    )


class StartInput(BaseModel):
    """Input for memory_start."""

    workflow_id: str | None = Field(
        default=None,
        description=(
            "ID of the workflow to follow. Pass the workflow_id from "
            "memory_search results, or omit to start without a workflow."
        ),
    )
    task_description: str | None = Field(
        default=None,
        description="Short description of the task being started.",
    )


# -- Tools -------------------------------------------------------------------


class MemorySearchTool(BaseTool):
    """Search for matching workflows in procedural memory."""

    name: str = "memory_search"
    description: str = (
        "Search for a procedure to complete the task. Call BEFORE starting any task. "
        "If a procedure is found, ask the user to confirm, then call memory_start(workflow_id=...). "
        "If no procedure is found, call memory_start(task_description=...) to capture the session."
    )
    args_schema: Type[BaseModel] = SearchInput

    _client: MyelinClient = PrivateAttr()
    _state: _MyelinToolState = PrivateAttr()

    def __init__(
        self,
        client: MyelinClient,
        state: _MyelinToolState,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._client = client
        self._state = state

    def _run(self, task_description: str) -> str:
        raise NotImplementedError("Use async: await memory_search.ainvoke(...)")

    async def _arun(self, task_description: str) -> str:
        try:
            r = await self._client.search(task_description)
        except Exception as exc:
            logger.warning("memory_search failed: %s", exc, exc_info=True)
            return f"Error: {exc}"

        self._state.last_search = r

        if r.top_match:
            m = r.top_match
            parts = [
                f"Top match (score: {m.score}):",
                f"  workflow_id: {m.workflow_id}",
                f"  title: {m.title}",
                f"  description: {m.description}",
                "",
                m.content or "",
                "",
            ]
            if r.other_matches:
                parts.append("Other matches:")
                for om in r.other_matches:
                    parts.append(
                        f"  - {om.title} (score: {om.score}, "
                        f"workflow_id: {om.workflow_id})"
                    )
                parts.append("")
            parts.append(
                "To follow this workflow, call memory_start with the "
                "workflow_id. To start without a workflow, call "
                "memory_start with no workflow_id."
            )
            return "\n".join(parts)
        else:
            if r.other_matches:
                parts = ["No strong match. Other workflows:"]
                for om in r.other_matches:
                    parts.append(
                        f"  - {om.title}: {om.description} "
                        f"(workflow_id: {om.workflow_id})"
                    )
                parts.append(
                    "\nCall memory_start with a workflow_id to follow one, "
                    "or call memory_start without one to work freestyle."
                )
                return "\n".join(parts)
            return (
                "No matching workflows found.\n"
                "Call memory_start to begin a freestyle recording session."
            )


class MemoryStartTool(BaseTool):
    """Start a recording session, optionally following a workflow."""

    name: str = "memory_start"
    description: str = (
        "Begin recording this task. Pass workflow_id to follow a procedure, "
        "or task_description for a freestyle session. All subsequent tool calls "
        "are captured until memory_finish is called."
    )
    args_schema: Type[BaseModel] = StartInput

    _client: MyelinClient = PrivateAttr()
    _state: _MyelinToolState = PrivateAttr()

    def __init__(
        self,
        client: MyelinClient,
        state: _MyelinToolState,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._client = client
        self._state = state

    def _run(
        self,
        workflow_id: str | None = None,
        task_description: str | None = None,
    ) -> str:
        raise NotImplementedError("Use async: await memory_start.ainvoke(...)")

    async def _arun(
        self,
        workflow_id: str | None = None,
        task_description: str | None = None,
    ) -> str:
        try:
            r = await self._client.start(
                workflow_id=workflow_id,
                task_description=task_description,
            )
        except Exception as exc:
            logger.warning("memory_start failed: %s", exc, exc_info=True)
            return f"Error: {exc}"

        self._state.session_id = r.session_id
        self._state.matched_workflow_id = r.matched_workflow_id
        self._state.active = True

        parts = [f"session_id: {r.session_id}"]
        if r.matched_workflow_id:
            parts.append(f"matched_workflow_id: {r.matched_workflow_id}")
        parts.append("\nSession started. When done, call memory_finish.")
        return "\n".join(parts)


class MemoryFinishTool(BaseTool):
    """Finalize the session and queue it for evaluation."""

    name: str = "memory_finish"
    description: str = (
        "Finalize the recording session and queue it for evaluation. "
        "Always call this when done — even if the task failed."
    )

    _client: MyelinClient = PrivateAttr()
    _state: _MyelinToolState = PrivateAttr()

    def __init__(
        self,
        client: MyelinClient,
        state: _MyelinToolState,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._client = client
        self._state = state

    def _run(self) -> str:
        raise NotImplementedError("Use async: await memory_finish.ainvoke(...)")

    async def _arun(self) -> str:
        if not self._state.session_id:
            return "Error: no active session. Call memory_start first."

        if not self._state.active:
            return "Session already finished."

        try:
            r = await self._client.finish(self._state.session_id)
        except Exception as exc:
            logger.warning("memory_finish failed: %s", exc, exc_info=True)
            return f"Error: {exc}"

        self._state.active = False

        response: dict = {
            "session_id": r.session_id,
            "tool_calls_recorded": r.tool_calls_recorded,
            "status": r.status,
        }
        if r.warning:
            response["warning"] = r.warning
        if r.workflow_id:
            response["workflow_id"] = r.workflow_id

        return json.dumps(response)
