"""Tests for MyelinSession."""

from unittest.mock import AsyncMock, patch

import pytest

from myelin_sdk.session import MyelinSession
from myelin_sdk.types import (
    CaptureResponse,
    FinishResponse,
    HintResponse,
    HintsResponse,
    RecallResponse,
    WorkflowInfo,
)


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.capture.return_value = CaptureResponse(status="ok")
    client.finish.return_value = FinishResponse(
        session_id="ses_1", tool_calls_recorded=3, status="evaluated"
    )
    client.hint.return_value = HintResponse(
        session_id="ses_1", step_number=1, detail="Do the thing"
    )
    client.hints.return_value = HintsResponse(
        session_id="ses_1", hints={1: "First", 2: "Second"}
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

    async def test_capture_after_finish_raises(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        await session.finish()
        with pytest.raises(RuntimeError, match="already finished"):
            await session.capture("Bash", {}, "out")


class TestFinish:
    async def test_finish_delegates(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        resp = await session.finish()
        assert resp.status == "evaluated"
        assert resp.tool_calls_recorded == 3
        mock_client.finish.assert_awaited_once_with("ses_1")

    async def test_double_finish_raises(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        await session.finish()
        with pytest.raises(RuntimeError, match="already finished"):
            await session.finish()


class TestHint:
    async def test_hint_delegates(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        resp = await session.hint(1)
        assert resp.detail == "Do the thing"
        mock_client.hint.assert_awaited_once_with("ses_1", 1)

    async def test_hint_works_after_finish(self, mock_client, recall_hit):
        """Hint should still work even after finish (read-only)."""
        session = MyelinSession(mock_client, recall_hit)
        await session.finish()
        resp = await session.hint(1)
        assert resp.step_number == 1


class TestStart:
    async def test_creates_session(self):
        mock_client = AsyncMock()
        recall = RecallResponse(session_id="ses_new", matched=False)
        mock_client.recall.return_value = recall

        with patch("myelin_sdk.session.MyelinClient", return_value=mock_client):
            session = await MyelinSession.start("do stuff", api_key="my_key123")

        assert session.session_id == "ses_new"
        assert session._owns_client is True
        mock_client.recall.assert_awaited_once_with("do stuff", agent_id="default")

    async def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("MYELIN_API_KEY", "my_env_key")
        mock_client = AsyncMock()
        mock_client.recall.return_value = RecallResponse(
            session_id="ses_env", matched=False
        )

        with patch("myelin_sdk.session.MyelinClient", return_value=mock_client) as cls:
            session = await MyelinSession.start("task")

        cls.assert_called_once()
        assert cls.call_args.kwargs["api_key"] == "my_env_key"

    async def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("MYELIN_API_KEY", raising=False)
        with pytest.raises(ValueError, match="api_key required"):
            MyelinSession.start("task")

    async def test_context_manager_without_await(self):
        """async with MyelinSession.start(...) as session: — no await needed."""
        mock_client = AsyncMock()
        recall = RecallResponse(session_id="ses_cm", matched=False)
        mock_client.recall.return_value = recall

        with patch("myelin_sdk.session.MyelinClient", return_value=mock_client):
            async with MyelinSession.start("task", api_key="key") as session:
                assert session.session_id == "ses_cm"
        mock_client.finish.assert_awaited_once()


class TestContextManager:
    async def test_auto_finish_on_exit(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit, _owns_client=False)
        async with session:
            pass
        mock_client.finish.assert_awaited_once_with("ses_1")

    async def test_skip_finish_if_already_done(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit, _owns_client=False)
        async with session:
            await session.finish()
        # finish called once explicitly, not again on exit
        mock_client.finish.assert_awaited_once()

    async def test_closes_owned_client(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit, _owns_client=True)
        async with session:
            pass
        mock_client.close.assert_awaited_once()

    async def test_no_close_unowned_client(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit, _owns_client=False)
        async with session:
            pass
        mock_client.close.assert_not_awaited()

    async def test_auto_finish_is_best_effort(self, mock_client, recall_hit):
        mock_client.finish.side_effect = RuntimeError("network error")
        session = MyelinSession(mock_client, recall_hit, _owns_client=False)
        # Should not raise
        async with session:
            pass


class TestLangchainHandler:
    def test_returns_handler(self, mock_client, recall_hit, patch_langchain):
        from myelin_sdk.integrations.langchain.handler import MyelinCallbackHandler

        session = MyelinSession(mock_client, recall_hit)
        handler = session.langchain_handler()
        assert isinstance(handler, MyelinCallbackHandler)
        assert handler._session_id == recall_hit.session_id
        assert handler._client is mock_client

    def test_forwards_kwargs(self, mock_client, recall_hit, patch_langchain):
        hide_in = lambda d: d
        hide_out = lambda s: s
        session = MyelinSession(mock_client, recall_hit)
        handler = session.langchain_handler(hide_inputs=hide_in, hide_outputs=hide_out)
        assert handler._hide_inputs is hide_in
        assert handler._hide_outputs is hide_out


class TestSteps:
    async def test_yields_tuples_on_hit(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        result = [(n, d) async for n, d in session.steps()]
        assert result == [(1, "First"), (2, "Second")]
        mock_client.hints.assert_awaited_once_with("ses_1")

    async def test_empty_on_miss(self, mock_client, recall_miss):
        session = MyelinSession(mock_client, recall_miss)
        result = [(n, d) async for n, d in session.steps()]
        assert result == []
        mock_client.hints.assert_not_awaited()


class TestBuildSystemPrompt:
    async def test_returns_prompt_on_hit(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        prompt = await session.build_system_prompt()
        assert prompt is not None
        assert "proven procedure" in prompt
        assert "Step 1: First" in prompt
        assert "Step 2: Second" in prompt
        assert "overview" in prompt

    async def test_returns_none_on_miss(self, mock_client, recall_miss):
        session = MyelinSession(mock_client, recall_miss)
        prompt = await session.build_system_prompt()
        assert prompt is None

    async def test_prepends_preamble(self, mock_client, recall_hit):
        session = MyelinSession(mock_client, recall_hit)
        prompt = await session.build_system_prompt(preamble="You are a helper.")
        assert prompt.startswith("You are a helper.")
        assert "proven procedure" in prompt
