"""MyelinToolkit — unified entry point for autonomous LangChain agents."""

from __future__ import annotations

import logging
from typing import Any, Callable

from ...client import MyelinClient
from ...redact import RedactionConfig
from .handler import MyelinCallbackHandler
from .state import _MyelinToolState
from .tools import MemoryFeedbackTool, MemoryFinishTool, MemoryHintTool, MemoryRecallTool

logger = logging.getLogger(__name__)


class MyelinToolkit:
    """Produces LangChain tools + callback handler linked via shared state.

    The agent calls ``memory_recall``, ``memory_hint``, ``memory_finish``
    autonomously; the handler captures all other tool calls.

    Usage::

        async with MyelinToolkit(api_key="my_...") as tk:
            agent = create_react_agent(llm, tk.tools)
            result = await agent.ainvoke(
                {"input": "..."},
                config={"callbacks": [tk.handler]},
            )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://myelin.fly.dev",
        agent_id: str = "default",
        *,
        redaction: RedactionConfig | None = None,
        hide_inputs: Callable[[dict], dict] | None = None,
        hide_outputs: Callable[[str], str] | None = None,
    ):
        self._client = MyelinClient(
            api_key=api_key, base_url=base_url, redaction=redaction
        )
        self._state = _MyelinToolState()
        self._agent_id = agent_id

        self._recall_tool = MemoryRecallTool(
            client=self._client, state=self._state, agent_id=agent_id
        )
        self._hint_tool = MemoryHintTool(
            client=self._client, state=self._state
        )
        self._feedback_tool = MemoryFeedbackTool(
            client=self._client, state=self._state
        )
        self._finish_tool = MemoryFinishTool(
            client=self._client, state=self._state
        )

        self._handler = MyelinCallbackHandler(
            client=self._client,
            session_id="",  # placeholder — handler reads from _state
            state=self._state,
            hide_inputs=hide_inputs,
            hide_outputs=hide_outputs,
            redaction=redaction,
        )

    @property
    def tools(self) -> list[Any]:
        return [self._recall_tool, self._hint_tool, self._feedback_tool, self._finish_tool]

    @property
    def handler(self) -> MyelinCallbackHandler:
        return self._handler

    @property
    def state(self) -> _MyelinToolState:
        return self._state

    async def __aenter__(self) -> MyelinToolkit:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._state.active and self._state.session_id:
            try:
                await self._client.finish(self._state.session_id)
                self._state.active = False
            except Exception:
                logger.warning(
                    "Auto-finish failed for session %s",
                    self._state.session_id,
                    exc_info=True,
                )
        await self._client.close()
