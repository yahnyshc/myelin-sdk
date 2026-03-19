"""LangChain tools for autonomous Myelin search/record/finish."""

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
from ...errors import MyelinAPIError
from .state import _MyelinToolState

logger = logging.getLogger(__name__)


# -- Input schemas -----------------------------------------------------------


class SearchInput(BaseModel):
    """Input for search."""

    task_description: str = Field(
        description=(
            "Natural language description of the task. "
            "Keep procedural detail (steps, output format, counts); "
            "omit only variable values (file paths, names, IDs) "
            "that change between uses."
        )
    )


class RecordInput(BaseModel):
    """Input for record."""

    workflow_id: str | None = Field(
        default=None,
        description=(
            "ID of the workflow to follow. Pass the workflow_id from "
            "search results, or omit to start without a workflow."
        ),
    )
    task_description: str | None = Field(
        default=None,
        description="Short description of the task being started.",
    )


# -- Tools -------------------------------------------------------------------


class MemorySearchTool(BaseTool):
    """Search for matching workflows in procedural memory."""

    name: str = "search"
    description: str = (
        "Look up whether a proven procedure exists for a given task. "
        "Returns the best match with full content, plus summaries of other candidates. "
        "Use the returned workflow_id with record to follow a procedure."
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
        raise NotImplementedError("Use async: await search.ainvoke(...)")

    async def _arun(self, task_description: str) -> str:
        try:
            r = await self._client.search(task_description)
        except MyelinAPIError as exc:
            logger.warning("search failed: %s", exc.detail)
            return f"Error: {exc.detail}"
        except Exception:
            logger.warning("search failed", exc_info=True)
            return "Error: unexpected failure"

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
                "To follow this workflow, call record with the "
                "workflow_id. To start without a workflow, call "
                "record with no workflow_id."
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
                    "\nCall record with a workflow_id to follow one, "
                    "or call record without one to work freestyle."
                )
                return "\n".join(parts)
            return (
                "No matching workflows found.\n"
                "Call record to begin a freestyle recording session."
            )


class MemoryRecordTool(BaseTool):
    """Start a recording session, optionally following a workflow."""

    name: str = "record"
    description: str = (
        "Activate session recording and begin capturing tool calls. "
        "Pass workflow_id to follow a known procedure, or task_description "
        "to record from scratch. Tool calls are captured and a new procedure "
        "can be extracted from freestyle sessions afterward."
    )
    args_schema: Type[BaseModel] = RecordInput

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
        raise NotImplementedError("Use async: await record.ainvoke(...)")

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
        except MyelinAPIError as exc:
            logger.warning("record failed: %s", exc.detail)
            return f"Error: {exc.detail}"
        except Exception:
            logger.warning("record failed", exc_info=True)
            return "Error: unexpected failure"

        self._state.session_id = r.session_id
        self._state.matched_workflow_id = r.matched_workflow_id
        self._state.active = True

        parts = [f"session_id: {r.session_id}"]
        if r.matched_workflow_id:
            parts.append(f"matched_workflow_id: {r.matched_workflow_id}")
        parts.append("\nSession started. When done, call finish.")
        return "\n".join(parts)


class MemoryFinishTool(BaseTool):
    """Finalize the session and queue it for evaluation."""

    name: str = "finish"
    description: str = (
        "Close the recording session and queue it for evaluation. "
        "Call this when the task is complete, whether it succeeded or failed."
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
        raise NotImplementedError("Use async: await finish.ainvoke(...)")

    async def _arun(self) -> str:
        if not self._state.session_id:
            return "Error: no active session. Call record first."

        if not self._state.active:
            return "Session already finished."

        try:
            r = await self._client.finish(self._state.session_id)
        except MyelinAPIError as exc:
            logger.warning("finish failed: %s", exc.detail)
            return f"Error: {exc.detail}"
        except Exception:
            logger.warning("finish failed", exc_info=True)
            return "Error: unexpected failure"

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
