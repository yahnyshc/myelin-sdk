"""LangChain callback handler for capturing tool calls to Myelin."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID

try:
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:
    raise ImportError(
        "Install langchain support: pip install myelin-sdk[langchain]"
    )

from ..._utils import truncate
from ...client import MyelinClient
from ...redact import RedactionConfig, redact_dict, redact_string

if TYPE_CHECKING:
    from .state import _MyelinToolState

logger = logging.getLogger(__name__)

_SKIP_TOOLS: frozenset[str] = frozenset({
    "memory_search",
    "memory_start",
    "memory_hint",
    "memory_finish",
})


class MyelinCallbackHandler(AsyncCallbackHandler):
    def __init__(
        self,
        client: MyelinClient,
        session_id: str,
        *,
        state: _MyelinToolState | None = None,
        hide_inputs: Callable[[dict], dict] | None = None,
        hide_outputs: Callable[[str], str] | None = None,
        redaction: RedactionConfig | None = None,
    ):
        self._client = client
        self._session_id = session_id
        self._state = state
        self._reasoning: dict[UUID, str] = {}
        self._pending_tools: dict[UUID, dict] = {}
        self._redaction = redaction

        # Compose redaction with user callbacks: redaction first, then user cb
        if redaction and redaction.enabled and redaction.redact_tool_input:
            base_hide_inputs = hide_inputs
            def _composed_hide_inputs(data: dict) -> dict:
                data = redact_dict(data, redaction)
                if base_hide_inputs:
                    data = base_hide_inputs(data)
                return data
            self._hide_inputs = _composed_hide_inputs
        else:
            self._hide_inputs = hide_inputs

        if redaction and redaction.enabled and redaction.redact_tool_response:
            base_hide_outputs = hide_outputs
            def _composed_hide_outputs(text: str) -> str:
                text = redact_string(text, redaction)
                if base_hide_outputs:
                    text = base_hide_outputs(text)
                return text
            self._hide_outputs = _composed_hide_outputs
        else:
            self._hide_outputs = hide_outputs

    @property
    def _effective_session_id(self) -> str | None:
        """Return session ID from shared state (dynamic) or constructor (static)."""
        if self._state is not None and self._state.session_id:
            return self._state.session_id
        return self._session_id or None

    async def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs: Any
    ) -> None:
        try:
            parts = []
            for gen_list in response.generations:
                for gen in gen_list:
                    if gen.text:
                        parts.append(gen.text)
            if parts:
                self._reasoning[run_id] = "\n".join(parts)
        except Exception:
            logger.debug(
                "Failed to extract reasoning from LLM response", exc_info=True
            )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")

        if tool_name in _SKIP_TOOLS:
            return

        try:
            tool_input = (
                json.loads(input_str) if isinstance(input_str, str) else input_str
            )
        except (json.JSONDecodeError, TypeError):
            tool_input = {"input": input_str}

        self._pending_tools[run_id] = {
            "name": tool_name,
            "input": tool_input,
            "parent_run_id": parent_run_id,
        }

    async def on_tool_end(
        self, output: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        await self._capture_tool(run_id, truncate(str(output)))

    async def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        await self._capture_tool(run_id, truncate(f"ERROR: {error}"))

    async def _capture_tool(self, run_id: UUID, output: str) -> None:
        pending = self._pending_tools.pop(run_id, None)
        if not pending:
            return

        session_id = self._effective_session_id
        if not session_id:
            return

        tool_input = pending["input"]
        tool_response = output

        if self._hide_inputs:
            tool_input = self._hide_inputs(tool_input)
        if self._hide_outputs:
            tool_response = self._hide_outputs(tool_response)

        reasoning = None
        parent_id = pending.get("parent_run_id")
        if parent_id:
            reasoning = self._reasoning.pop(parent_id, None)

        if (
            reasoning
            and self._redaction
            and self._redaction.enabled
            and self._redaction.redact_reasoning
        ):
            reasoning = redact_string(reasoning, self._redaction)

        try:
            await self._client.capture(
                session_id=session_id,
                tool_name=pending["name"],
                tool_input=tool_input,
                tool_response=tool_response,
                reasoning=reasoning,
                client_ts=time.time(),
            )
        except Exception:
            logger.warning(
                "Failed to capture tool call %s", pending["name"], exc_info=True
            )
