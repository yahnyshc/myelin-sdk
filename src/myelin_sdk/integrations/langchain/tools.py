"""LangChain tools for autonomous Myelin recall/hint/finish."""

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


class RecallInput(BaseModel):
    """Input for memory_recall."""

    task_description: str = Field(
        description=(
            "Natural language description of the task. "
            "Keep procedural detail (steps, output format, counts); "
            "omit only variable values (file paths, names, IDs) "
            "that change between uses."
        )
    )


class FeedbackInput(BaseModel):
    """Input for memory_feedback."""

    notes: str = Field(
        description=(
            "Your observation or note about the session — what worked, what failed, "
            "error messages, or any context about the task."
        )
    )


class HintInput(BaseModel):
    """Input for memory_hint."""

    step_number: int = Field(
        description="The step number to get detail for."
    )


# -- Tools -------------------------------------------------------------------


class MemoryRecallTool(BaseTool):
    """Search for a matching workflow and start a recording session."""

    name: str = "memory_recall"
    description: str = (
        "Search for a matching workflow and start a recording session. "
        "On HIT: returns workflow steps + session_id. Follow the workflow, "
        "then call memory_finish. "
        "On MISS: returns session_id. Work freestyle, then call memory_finish when done."
    )
    args_schema: Type[BaseModel] = RecallInput

    _client: MyelinClient = PrivateAttr()
    _state: _MyelinToolState = PrivateAttr()
    _agent_id: str = PrivateAttr()

    def __init__(
        self,
        client: MyelinClient,
        state: _MyelinToolState,
        agent_id: str = "default",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._client = client
        self._state = state
        self._agent_id = agent_id

    def _run(self, task_description: str) -> str:
        raise NotImplementedError("Use async: await memory_recall.ainvoke(...)")

    async def _arun(self, task_description: str) -> str:
        try:
            r = await self._client.recall(task_description, self._agent_id)
        except Exception as exc:
            logger.warning("memory_recall failed: %s", exc, exc_info=True)
            return f"Error: {exc}"

        self._state.session_id = r.session_id
        self._state.matched = r.matched
        self._state.workflow = r.workflow
        self._state.active = True

        if r.matched and r.workflow:
            wf = r.workflow
            steps_block = "\n".join(
                f"{i + 1}) {step}" for i, step in enumerate(wf.steps)
            )
            return (
                f"session_id: {r.session_id}\n"
                f"\n"
                f"{wf.overview}\n"
                f"\n"
                f"Steps:\n"
                f"{steps_block}\n"
                f"\n"
                f"Call memory_hint(step_number) for detail on any step.\n"
                f"Use memory_feedback(notes) to record observations during execution.\n"
                f"When done, call memory_finish."
            )
        else:
            return (
                f"session_id: {r.session_id}\n"
                f"\n"
                f"No matching workflow found. Work freestyle.\n"
                f"Use memory_feedback(notes) to record observations during execution.\n"
                f"When done, call memory_finish."
            )


class MemoryHintTool(BaseTool):
    """Get detailed instructions for a single workflow step."""

    name: str = "memory_hint"
    description: str = (
        "Get detailed instructions for a single workflow step. "
        "Call this during a HIT workflow when you need more detail on a specific step."
    )
    args_schema: Type[BaseModel] = HintInput

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

    def _run(self, step_number: int) -> str:
        raise NotImplementedError("Use async: await memory_hint.ainvoke(...)")

    async def _arun(self, step_number: int) -> str:
        if not self._state.session_id:
            return "Error: no active session. Call memory_recall first."

        try:
            r = await self._client.hint(self._state.session_id, step_number)
        except Exception as exc:
            logger.warning("memory_hint failed: %s", exc, exc_info=True)
            return f"Error: {exc}"

        return json.dumps(
            {
                "session_id": r.session_id,
                "step_number": r.step_number,
                "detail": r.detail,
            }
        )


class MemoryFeedbackTool(BaseTool):
    """Write a note about the current session."""

    name: str = "memory_feedback"
    description: str = (
        "Write a note about the current session — what worked, what failed, "
        "error messages, or any context about the task. "
        "Notes are stored and used for session evaluation."
    )
    args_schema: Type[BaseModel] = FeedbackInput

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

    def _run(self, notes: str) -> str:
        raise NotImplementedError("Use async: await memory_feedback.ainvoke(...)")

    async def _arun(self, notes: str) -> str:
        if not self._state.session_id:
            return "Error: no active session. Call memory_recall first."

        try:
            r = await self._client.feedback(self._state.session_id, notes)
        except Exception as exc:
            logger.warning("memory_feedback failed: %s", exc, exc_info=True)
            return f"Error: {exc}"

        return json.dumps({
            "session_id": r.session_id,
            "status": r.status,
        })


class MemoryFinishTool(BaseTool):
    """Finalize the session and queue it for evaluation."""

    name: str = "memory_finish"
    description: str = (
        "Finalize the current session and queue it for server-side evaluation. "
        "Call this after recall, once the task is done. "
        "The server evaluates the session independently using an LLM judge."
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
            return "Error: no active session. Call memory_recall first."

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
