"""Tests for MyelinSession."""

from unittest.mock import AsyncMock

import pytest

from myelin_sdk.session import MyelinSession
from myelin_sdk.types import (
    CaptureResponse,
    DebriefResponse,
    HintResponse,
    RecallResponse,
    WorkflowInfo,
)


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.capture.return_value = CaptureResponse(status="ok")
    client.debrief.return_value = DebriefResponse(
        session_id="ses_1", tool_calls_recorded=3, status="evaluated"
    )
    client.hint.return_value = HintResponse(
        session_id="ses_1", step_number=1, detail="Do the thing"
    )
    return client


@pytest.fixture
def recall_hit():
    return RecallResponse(
        session_id="ses_1",
        matched=True,
        workflow=WorkflowInfo(
            id="wf_1",
            description="test",
            total_steps=2,
            overview="overview",
            skeleton="skeleton",
        ),
    )


@pytest.fixture
def recall_miss():
    return RecallResponse(session_id="ses_2", matched=False)


class TestProperties:
    def test_session_id(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        assert session.session_id == "ses_1"

    def test_matched_true(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        assert session.matched is True

    def test_matched_false(self, mock_client, recall_miss):
        session = MyelinSession(mock_client, recall_miss)
        assert session.matched is False

    def test_workflow_present(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        assert session.workflow is not None
        assert session.workflow.id == "wf_1"

    def test_workflow_none(self, mock_client, recall_miss):
        session = MyelinSession(mock_client, recall_miss)
        assert session.workflow is None


class TestCapture:
    async def test_capture_delegates(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        resp = await session.capture("Bash", {"cmd": "ls"}, "output")
        assert resp.status == "ok"
        mock_client.capture.assert_awaited_once_with(
            "ses_1", "Bash", {"cmd": "ls"}, "output", None, None
        )

    async def test_capture_with_optional_args(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        await session.capture(
            "Read", {"path": "/f"}, "data", reasoning="why", client_ts=100.0
        )
        mock_client.capture.assert_awaited_once_with(
            "ses_1", "Read", {"path": "/f"}, "data", "why", 100.0
        )

    async def test_capture_after_debrief_raises(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        await session.debrief()
        with pytest.raises(RuntimeError, match="already debriefed"):
            await session.capture("Bash", {}, "out")


class TestDebrief:
    async def test_debrief_delegates(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        resp = await session.debrief()
        assert resp.status == "evaluated"
        assert resp.tool_calls_recorded == 3
        mock_client.debrief.assert_awaited_once_with("ses_1")

    async def test_double_debrief_raises(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        await session.debrief()
        with pytest.raises(RuntimeError, match="already debriefed"):
            await session.debrief()


class TestHint:
    async def test_hint_delegates(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        resp = await session.hint(1)
        assert resp.detail == "Do the thing"
        mock_client.hint.assert_awaited_once_with("ses_1", 1)

    async def test_hint_works_after_debrief(self, mock_client, recall_hit):
        """Hint should still work even after debrief (read-only)."""
        session = MyelinSession(mock_client, recall_hit)
        await session.debrief()
        resp = await session.hint(1)
        assert resp.step_number == 1
