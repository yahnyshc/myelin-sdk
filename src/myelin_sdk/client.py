"""Async HTTP client for the Myelin REST API."""

import time

import httpx

from importlib.metadata import version as _pkg_version

from .errors import raise_for_status
from .redact import RedactionConfig, get_default_config, redact_dict, redact_string
from .types import CaptureResponse, FeedbackResponse, FinishResponse, ListWorkflowsResponse, RecallResponse

_VERSION = _pkg_version("myelin-sdk")


class MyelinClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://myelin.fly.dev",
        *,
        redaction: RedactionConfig | None = None,
    ):

        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": f"myelin-sdk/{_VERSION}",
            },
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        self._redaction: RedactionConfig = (
            redaction if redaction is not None else get_default_config()
        )

    async def list_workflows(self) -> ListWorkflowsResponse:
        resp = await self._http.get("/v1/workflows")
        raise_for_status(resp)
        return ListWorkflowsResponse(**resp.json())

    async def recall(
        self, task_description: str, agent_id: str = "default",
        workflow_id: str | None = None,
    ) -> RecallResponse:
        payload: dict = {"task_description": task_description, "agent_id": agent_id}
        if workflow_id:
            payload["workflow_id"] = workflow_id
        resp = await self._http.post("/v1/recall", json=payload)
        raise_for_status(resp)
        return RecallResponse(**resp.json())

    async def capture(
        self,
        session_id: str,
        tool_name: str,
        tool_input: dict,
        tool_response: str,
        reasoning: str | None = None,
        client_ts: float | None = None,
    ) -> CaptureResponse:
        if self._redaction.enabled:
            if self._redaction.redact_tool_input:
                tool_input = redact_dict(tool_input, self._redaction)
            if self._redaction.redact_tool_response:
                tool_response = redact_string(
                    str(tool_response), self._redaction
                )
            if reasoning and self._redaction.redact_reasoning:
                reasoning = redact_string(reasoning, self._redaction)

        payload: dict = {
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": tool_response,
            "client_ts": client_ts or time.time(),
        }
        if reasoning:
            payload["reasoning"] = reasoning
        resp = await self._http.post("/v1/capture", json=payload)
        raise_for_status(resp)
        return CaptureResponse(**resp.json())

    async def feedback(self, session_id: str, notes: str) -> FeedbackResponse:
        """Append a timestamped note to the session. Call often, not just at the end."""
        resp = await self._http.post(
            f"/v1/sessions/{session_id}/feedback",
            json={"notes": notes},
        )
        raise_for_status(resp)
        return FeedbackResponse(**resp.json())

    async def finish(self, session_id: str) -> FinishResponse:
        resp = await self._http.post(f"/v1/sessions/{session_id}/finish")
        raise_for_status(resp)
        return FinishResponse(**resp.json())

    async def close(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
