"""Tests for MyelinSession."""

from unittest.mock import AsyncMock, patch

import pytest

from myelin_sdk.session import MyelinSession
from myelin_sdk.types import (
    CaptureResponse,
    FinishResponse,
    StartResult,
)


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.capture.return_value = CaptureResponse(status="ok")
    client.finish.return_value = FinishResponse(
        session_id="ses_1", tool_calls_recorded=3, status="evaluated"
    )
    return client


@pytest.fixture
def start_with_workflow():
    return StartResult(
        session_id="ses_1",
        matched_workflow_id="wf_1",
    )


@pytest.fixture
def start_without_workflow():
    return StartResult(session_id="ses_2", matched_workflow_id=None)


class TestProperties:
    def test_session_id(self, mock_client, start_with_workflow):
        session = MyelinSession(mock_client, start_with_workflow)
        assert session.session_id == "ses_1"

    def test_matched_workflow_id(self, mock_client, start_with_workflow):
        session = MyelinSession(mock_client, start_with_workflow)
        assert session.matched_workflow_id == "wf_1"

    def test_no_matched_workflow(self, mock_client, start_without_workflow):
        session = MyelinSession(mock_client, start_without_workflow)
        assert session.matched_workflow_id is None


class TestCapture:
    async def test_capture_delegates(self, mock_client, start_with_workflow):
        session = MyelinSession(mock_client, start_with_workflow)
        resp = await session.capture("Bash", {"cmd": "ls"}, "output")
        assert resp.status == "ok"
        mock_client.capture.assert_awaited_once_with(
            "ses_1", "Bash", {"cmd": "ls"}, "output", None, None
        )

    async def test_capture_with_optional_args(self, mock_client, start_with_workflow):
        session = MyelinSession(mock_client, start_with_workflow)
        await session.capture(
            "Read", {"path": "/f"}, "data", reasoning="why", client_ts=100.0
        )
        mock_client.capture.assert_awaited_once_with(
            "ses_1", "Read", {"path": "/f"}, "data", "why", 100.0
        )

    async def test_capture_after_finish_raises(self, mock_client, start_with_workflow):
        session = MyelinSession(mock_client, start_with_workflow)
        await session.finish()
        with pytest.raises(RuntimeError, match="already finished"):
            await session.capture("Bash", {}, "out")


class TestFinish:
    async def test_finish_delegates(self, mock_client, start_with_workflow):
        session = MyelinSession(mock_client, start_with_workflow)
        resp = await session.finish()
        assert resp.status == "evaluated"
        assert resp.tool_calls_recorded == 3
        mock_client.finish.assert_awaited_once_with("ses_1")

    async def test_double_finish_raises(self, mock_client, start_with_workflow):
        session = MyelinSession(mock_client, start_with_workflow)
        await session.finish()
        with pytest.raises(RuntimeError, match="already finished"):
            await session.finish()


class TestCreate:
    async def test_creates_session(self):
        mock_client = AsyncMock()
        start_result = StartResult(session_id="ses_new", matched_workflow_id=None)
        mock_client.start.return_value = start_result

        with patch("myelin_sdk.session.MyelinClient", return_value=mock_client):
            session = await MyelinSession.create("do stuff", api_key="my_key123")

        assert session.session_id == "ses_new"
        assert session._owns_client is True
        mock_client.start.assert_awaited_once_with(
            workflow_id=None, task_description="do stuff", project_id=None
        )

    async def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("MYELIN_API_KEY", "my_env_key")
        mock_client = AsyncMock()
        mock_client.start.return_value = StartResult(
            session_id="ses_env", matched_workflow_id=None
        )

        with patch("myelin_sdk.session.MyelinClient", return_value=mock_client) as cls:
            await MyelinSession.create("task")

        cls.assert_called_once()
        assert cls.call_args.kwargs["api_key"] == "my_env_key"

    async def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("MYELIN_API_KEY", raising=False)
        with pytest.raises(ValueError, match="api_key required"):
            MyelinSession.create("task")

    async def test_context_manager_without_await(self):
        """async with MyelinSession.create(...) as session: — no await needed."""
        mock_client = AsyncMock()
        start_result = StartResult(session_id="ses_cm", matched_workflow_id=None)
        mock_client.start.return_value = start_result

        with patch("myelin_sdk.session.MyelinClient", return_value=mock_client):
            async with MyelinSession.create("task", api_key="key") as session:
                assert session.session_id == "ses_cm"
        mock_client.finish.assert_awaited_once()

    async def test_create_with_workflow_id(self):
        mock_client = AsyncMock()
        start_result = StartResult(
            session_id="ses_wf", matched_workflow_id="wf_1"
        )
        mock_client.start.return_value = start_result

        with patch("myelin_sdk.session.MyelinClient", return_value=mock_client):
            session = await MyelinSession.create(
                "deploy", api_key="key", workflow_id="wf_1"
            )

        assert session.session_id == "ses_wf"
        assert session.matched_workflow_id == "wf_1"
        mock_client.start.assert_awaited_once_with(
            workflow_id="wf_1", task_description="deploy", project_id=None
        )


class TestContextManager:
    async def test_auto_finish_on_exit(self, mock_client, start_with_workflow):
        session = MyelinSession(mock_client, start_with_workflow, _owns_client=False)
        async with session:
            pass
        mock_client.finish.assert_awaited_once_with("ses_1")

    async def test_skip_finish_if_already_done(self, mock_client, start_with_workflow):
        session = MyelinSession(mock_client, start_with_workflow, _owns_client=False)
        async with session:
            await session.finish()
        # finish called once explicitly, not again on exit
        mock_client.finish.assert_awaited_once()

    async def test_closes_owned_client(self, mock_client, start_with_workflow):
        session = MyelinSession(mock_client, start_with_workflow, _owns_client=True)
        async with session:
            pass
        mock_client.close.assert_awaited_once()

    async def test_no_close_unowned_client(self, mock_client, start_with_workflow):
        session = MyelinSession(mock_client, start_with_workflow, _owns_client=False)
        async with session:
            pass
        mock_client.close.assert_not_awaited()

    async def test_auto_finish_is_best_effort(self, mock_client, start_with_workflow):
        mock_client.finish.side_effect = RuntimeError("network error")
        session = MyelinSession(mock_client, start_with_workflow, _owns_client=False)
        # Should not raise
        async with session:
            pass


class TestLangchainHandler:
    def test_returns_handler(self, mock_client, start_with_workflow, patch_langchain):
        from myelin_sdk.integrations.langchain.handler import MyelinCallbackHandler

        session = MyelinSession(mock_client, start_with_workflow)
        handler = session.langchain_handler()
        assert isinstance(handler, MyelinCallbackHandler)
        assert handler._session_id == start_with_workflow.session_id
        assert handler._client is mock_client

    def test_forwards_kwargs(self, mock_client, start_with_workflow, patch_langchain):
        def hide_in(d): return d
        def hide_out(s): return s
        session = MyelinSession(mock_client, start_with_workflow)
        handler = session.langchain_handler(hide_inputs=hide_in, hide_outputs=hide_out)
        assert handler._hide_inputs is hide_in
        assert handler._hide_outputs is hide_out
