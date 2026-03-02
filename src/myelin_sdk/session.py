"""Session state wrapper for Myelin interactions."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from .client import MyelinClient
from .redact import RedactionConfig
from .types import CaptureResponse, FeedbackResponse, FinishResponse, HintResponse, RecallResponse, WorkflowInfo

if TYPE_CHECKING:
    from typing import Callable


class MyelinSession:
    def __init__(
        self,
        client: MyelinClient,
        recall_response: RecallResponse,
        *,
        _owns_client: bool = False,
    ):
        self._client = client
        self._recall = recall_response
        self._active = True
        self._owns_client = _owns_client
        self._init_coro = None

    @classmethod
    def start(
        cls,
        task: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://myelin.fly.dev",
        agent_id: str = "default",
        redaction: RedactionConfig | None = None,
    ) -> MyelinSession:
        key = api_key or os.environ.get("MYELIN_API_KEY")
        if not key:
            raise ValueError("api_key required (pass it or set MYELIN_API_KEY)")

        client = MyelinClient(api_key=key, base_url=base_url, redaction=redaction)
        # Create a placeholder with a sentinel recall response; __await__ fills it in.
        placeholder = RecallResponse(session_id="", matched=False)
        session = cls(client, placeholder, _owns_client=True)

        async def _init() -> MyelinSession:
            session._recall = await client.recall(task, agent_id=agent_id)
            return session

        session._init_coro = _init()
        return session

    def __await__(self):
        return self._init_coro.__await__()

    @property
    def session_id(self) -> str:
        return self._recall.session_id

    @property
    def matched(self) -> bool:
        return self._recall.matched

    @property
    def workflow(self) -> WorkflowInfo | None:
        return self._recall.workflow

    async def capture(
        self,
        tool_name: str,
        tool_input: dict,
        tool_response: str,
        reasoning: str | None = None,
        client_ts: float | None = None,
    ) -> CaptureResponse:
        if not self._active:
            raise RuntimeError("Session already finished")
        return await self._client.capture(
            self.session_id, tool_name, tool_input, tool_response, reasoning, client_ts
        )

    async def feedback(self, notes: str) -> FeedbackResponse:
        if not self._active:
            raise RuntimeError("Session already finished")
        return await self._client.feedback(self.session_id, notes)

    async def finish(self) -> FinishResponse:
        if not self._active:
            raise RuntimeError("Session already finished")
        result = await self._client.finish(self.session_id)
        self._active = False
        return result

    async def hint(self, step_number: int) -> HintResponse:
        return await self._client.hint(self.session_id, step_number)

    def langchain_handler(
        self,
        *,
        hide_inputs: Callable | None = None,
        hide_outputs: Callable | None = None,
        redaction: RedactionConfig | None = None,
    ):
        from .integrations.langchain.handler import MyelinCallbackHandler

        return MyelinCallbackHandler(
            client=self._client,
            session_id=self.session_id,
            hide_inputs=hide_inputs,
            hide_outputs=hide_outputs,
            redaction=redaction,
        )

    async def steps(self) -> AsyncIterator[tuple[int, str]]:
        if not self.matched or not self.workflow:
            return
        resp = await self._client.hints(self.session_id)
        for step_num in sorted(resp.hints):
            yield step_num, resp.hints[step_num]

    async def build_system_prompt(self, *, preamble: str = "") -> str | None:
        if not self.matched or not self.workflow:
            return None
        wf = self.workflow
        hints = [f"Step {n}: {d}" async for n, d in self.steps()]
        prompt = (
            f"You are following a proven procedure for: {wf.description}\n\n"
            f"Overview: {wf.overview}\n\n"
            f"Steps:\n" + "\n".join(hints) + "\n\n"
            "Adapt these steps to the specific request. "
            "Use the available tools to execute each step."
        )
        return preamble.rstrip() + "\n\n" + prompt if preamble else prompt

    async def __aenter__(self):
        if self._init_coro is not None:
            await self._init_coro
            self._init_coro = None
        return self

    async def __aexit__(self, *exc):
        if self._active:
            try:
                await self.finish()
            except Exception:
                pass
        if self._owns_client:
            await self._client.close()
