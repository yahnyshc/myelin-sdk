"""Session state wrapper for Myelin interactions."""

from .client import MyelinClient
from .types import CaptureResponse, DebriefResponse, HintResponse, RecallResponse


class MyelinSession:
    def __init__(self, client: MyelinClient, recall_response: RecallResponse):
        self._client = client
        self._recall = recall_response
        self._active = True

    @property
    def session_id(self) -> str:
        return self._recall.session_id

    @property
    def matched(self) -> bool:
        return self._recall.matched

    @property
    def workflow(self):
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
            raise RuntimeError("Session already debriefed")
        return await self._client.capture(
            self.session_id, tool_name, tool_input, tool_response, reasoning, client_ts
        )

    async def debrief(self) -> DebriefResponse:
        if not self._active:
            raise RuntimeError("Session already debriefed")
        result = await self._client.debrief(self.session_id)
        self._active = False
        return result

    async def hint(self, step_number: int) -> HintResponse:
        return await self._client.hint(self.session_id, step_number)
