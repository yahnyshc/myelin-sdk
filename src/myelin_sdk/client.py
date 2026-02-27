"""Async HTTP client for the Myelin REST API."""

import time

import httpx

from importlib.metadata import version as _pkg_version

from .types import CaptureResponse, DebriefResponse, HintResponse, RecallResponse

_VERSION = _pkg_version("myelin-sdk")


class MyelinClient:
    def __init__(self, api_key: str, base_url: str = "https://myelin.fly.dev"):
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": f"myelin-sdk/{_VERSION}",
            },
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=0),
        )

    async def recall(
        self, task_description: str, agent_id: str = "langchain"
    ) -> RecallResponse:
        resp = await self._http.post(
            "/v1/recall",
            json={"task_description": task_description, "agent_id": agent_id},
        )
        resp.raise_for_status()
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
        resp.raise_for_status()
        return CaptureResponse(**resp.json())

    async def debrief(self, session_id: str) -> DebriefResponse:
        resp = await self._http.post(f"/v1/sessions/{session_id}/debrief")
        resp.raise_for_status()
        return DebriefResponse(**resp.json())

    async def hint(self, session_id: str, step_number: int) -> HintResponse:
        resp = await self._http.get(
            f"/v1/sessions/{session_id}/hint/{step_number}"
        )
        resp.raise_for_status()
        return HintResponse(**resp.json())

    async def close(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
