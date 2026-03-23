"""Async HTTP client for the Myelin REST API."""

import time

import httpx

from importlib.metadata import version as _pkg_version

from ._utils import validate_base_url
from .errors import raise_for_status
from .redact import RedactionConfig, get_default_config, redact_dict, redact_string
from .types import (
    CaptureResponse,
    FeedbackResponse,
    FinishResponse,
    SearchResult,
    StartResult,
    SyncResult,
)

_VERSION = _pkg_version("myelin-sdk")


class MyelinClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://myelin.fly.dev",
        *,
        redaction: RedactionConfig | None = None,
    ):

        validate_base_url(base_url)
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

    async def search(
        self,
        task_description: str | None = None,
        project_id: str | None = None,
    ) -> SearchResult:
        payload: dict = {}
        if task_description:
            payload["task_description"] = task_description
        if project_id:
            payload["project_id"] = project_id
        resp = await self._http.post("/v1/search", json=payload)
        raise_for_status(resp)
        return SearchResult(**resp.json())

    async def start(
        self,
        workflow_id: str | None = None,
        task_description: str | None = None,
        project_id: str | None = None,
    ) -> StartResult:
        payload: dict = {}
        if workflow_id:
            payload["workflow_id"] = workflow_id
        if task_description:
            payload["task_description"] = task_description
        if project_id:
            payload["project_id"] = project_id
        resp = await self._http.post("/v1/start", json=payload)
        raise_for_status(resp)
        return StartResult(**resp.json())

    async def capture(
        self,
        session_id: str,
        tool_name: str,
        tool_input: dict,
        tool_response: str,
        context: str | None = None,
        client_ts: float | None = None,
    ) -> CaptureResponse:
        if self._redaction.enabled:
            if self._redaction.redact_tool_input:
                tool_input = redact_dict(tool_input, self._redaction)
            if self._redaction.redact_tool_response:
                tool_response = redact_string(
                    str(tool_response), self._redaction
                )
            if context and self._redaction.redact_context:
                context = redact_string(context, self._redaction)

        payload: dict = {
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": tool_response,
            "client_ts": client_ts or time.time(),
        }
        if context:
            payload["context"] = context
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

    async def sync_workflows(
        self,
        files: list[dict],
    ) -> SyncResult:
        """Sync local markdown procedure files to the Myelin server.

        Each entry in *files* must have:
          - ``path``: relative file path (used as identifier)
          - ``content``: full markdown content
          - ``description``: short description (from first heading or filename)
        """
        resp = await self._http.post(
            "/v1/workflows/sync",
            json={"files": files},
        )
        raise_for_status(resp)
        return SyncResult(**resp.json())

    async def close(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
