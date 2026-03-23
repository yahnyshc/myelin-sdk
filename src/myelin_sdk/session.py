"""Session state wrapper for Myelin interactions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .client import MyelinClient
from .redact import RedactionConfig
from .types import CaptureResponse, FeedbackResponse, FinishResponse, StartResult

if TYPE_CHECKING:
    from typing import Callable


class MyelinSession:
    def __init__(
        self,
        client: MyelinClient,
        start_result: StartResult,
        *,
        _owns_client: bool = False,
    ):
        self._client = client
        self._start_result = start_result
        self._active = True
        self._owns_client = _owns_client
        self._init_coro = None

    @classmethod
    def create(
        cls,
        task: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://myelin.fly.dev",
        workflow_id: str | None = None,
        project_id: str | None = None,
        redaction: RedactionConfig | None = None,
    ) -> MyelinSession:
        key = api_key or os.environ.get("MYELIN_API_KEY")
        if not key:
            raise ValueError("api_key required (pass it or set MYELIN_API_KEY)")

        client = MyelinClient(api_key=key, base_url=base_url, redaction=redaction)
        # Create a placeholder; __await__ / __aenter__ fills it in.
        placeholder = StartResult(session_id="")
        session = cls(client, placeholder, _owns_client=True)

        async def _init() -> MyelinSession:
            session._start_result = await client.start(
                workflow_id=workflow_id,
                task_description=task,
                project_id=project_id,
            )
            return session

        session._init_coro = _init()
        return session

    def __await__(self):
        return self._init_coro.__await__()

    @property
    def session_id(self) -> str:
        return self._start_result.session_id

    @property
    def matched_workflow_id(self) -> str | None:
        return self._start_result.matched_workflow_id

    async def capture(
        self,
        tool_name: str,
        tool_input: dict,
        tool_response: str,
        context: str | None = None,
        client_ts: float | None = None,
    ) -> CaptureResponse:
        if not self._active:
            raise RuntimeError("Session already finished")
        return await self._client.capture(
            self.session_id, tool_name, tool_input, tool_response, context, client_ts
        )

    async def feedback(self, notes: str) -> FeedbackResponse:
        """Jot down what just happened — a decision, error, or discovery.

        Call this throughout the session whenever something noteworthy occurs,
        not as a summary at the end. Each call appends a separate timestamped
        note. Short and frequent beats long and once.
        """
        if not self._active:
            raise RuntimeError("Session already finished")
        return await self._client.feedback(self.session_id, notes)

    async def finish(self) -> FinishResponse:
        if not self._active:
            raise RuntimeError("Session already finished")
        result = await self._client.finish(self.session_id)
        self._active = False
        return result

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
