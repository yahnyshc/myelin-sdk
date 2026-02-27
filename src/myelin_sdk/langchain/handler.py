"""LangChain callback handler for capturing tool calls to Myelin."""

import json
import logging
import time
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:
    raise ImportError(
        "Install langchain support: pip install myelin-sdk[langchain]"
    )

from ..client import MyelinClient

logger = logging.getLogger(__name__)

MAX_RESPONSE_LEN = 8000
_HEAD = MAX_RESPONSE_LEN // 2
_TAIL = MAX_RESPONSE_LEN // 2


def _truncate(text: str) -> str:
    if len(text) <= MAX_RESPONSE_LEN:
        return text
    return (
        text[:_HEAD]
        + f"\n... [{len(text)} chars, middle truncated] ...\n"
        + text[-_TAIL:]
    )


class MyelinCallbackHandler(AsyncCallbackHandler):
    def __init__(self, client: MyelinClient, session_id: str):
        self._client = client
        self._session_id = session_id
        self._reasoning: dict[UUID, str] = {}
        self._pending_tools: dict[UUID, dict] = {}

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
        await self._capture_tool(run_id, _truncate(str(output)))

    async def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        await self._capture_tool(run_id, _truncate(f"ERROR: {error}"))

    async def _capture_tool(self, run_id: UUID, output: str) -> None:
        pending = self._pending_tools.pop(run_id, None)
        if not pending:
            return

        reasoning = None
        parent_id = pending.get("parent_run_id")
        if parent_id:
            reasoning = self._reasoning.pop(parent_id, None)

        try:
            await self._client.capture(
                session_id=self._session_id,
                tool_name=pending["name"],
                tool_input=pending["input"],
                tool_response=output,
                reasoning=reasoning,
                client_ts=time.time(),
            )
        except Exception:
            logger.warning(
                "Failed to capture tool call %s", pending["name"], exc_info=True
            )
