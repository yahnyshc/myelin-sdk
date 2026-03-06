"""Tests for MyelinToolkit — lifecycle, tool creation, handler integration."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def _patch_langchain(patch_langchain):
    """Use the shared langchain patch from conftest."""


def _mock_client():
    client = AsyncMock()
    client.recall = AsyncMock()
    client.capture = AsyncMock()
    client.finish = AsyncMock()
    client.close = AsyncMock()
    return client


class TestToolkitCreation:
    def test_creates_tools(self):
        from myelin_sdk.integrations.langchain.toolkit import MyelinToolkit

        tk = MyelinToolkit(api_key="test_key")
        assert len(tk.tools) == 2
        names = {t.name for t in tk.tools}
        assert names == {"memory_recall", "memory_finish"}

    def test_handler_is_created(self):
        from myelin_sdk.integrations.langchain.handler import MyelinCallbackHandler
        from myelin_sdk.integrations.langchain.toolkit import MyelinToolkit

        tk = MyelinToolkit(api_key="test_key")
        assert isinstance(tk.handler, MyelinCallbackHandler)

    def test_shared_state(self):
        from myelin_sdk.integrations.langchain.toolkit import MyelinToolkit

        tk = MyelinToolkit(api_key="test_key")
        # All tools and handler share the same state object
        state = tk.state
        for tool in tk.tools:
            assert tool._state is state
        assert tk.handler._state is state


class TestToolkitLifecycle:
    async def test_context_manager_closes_client(self):
        from myelin_sdk.integrations.langchain.toolkit import MyelinToolkit

        tk = MyelinToolkit(api_key="test_key")
        mock_client = _mock_client()
        tk._client = mock_client

        async with tk:
            pass

        mock_client.close.assert_awaited_once()

    async def test_auto_finish_on_exit(self):
        from myelin_sdk.integrations.langchain.toolkit import MyelinToolkit

        tk = MyelinToolkit(api_key="test_key")
        mock_client = _mock_client()
        tk._client = mock_client

        # Simulate an active session
        tk._state.session_id = "ses_active"
        tk._state.active = True

        async with tk:
            pass

        mock_client.finish.assert_awaited_once_with("ses_active")
        assert tk._state.active is False
        mock_client.close.assert_awaited_once()

    async def test_no_auto_finish_when_already_finished(self):
        from myelin_sdk.integrations.langchain.toolkit import MyelinToolkit

        tk = MyelinToolkit(api_key="test_key")
        mock_client = _mock_client()
        tk._client = mock_client

        tk._state.session_id = "ses_done"
        tk._state.active = False

        async with tk:
            pass

        mock_client.finish.assert_not_awaited()
        mock_client.close.assert_awaited_once()

    async def test_no_auto_finish_when_no_session(self):
        from myelin_sdk.integrations.langchain.toolkit import MyelinToolkit

        tk = MyelinToolkit(api_key="test_key")
        mock_client = _mock_client()
        tk._client = mock_client

        async with tk:
            pass

        mock_client.finish.assert_not_awaited()

    async def test_auto_finish_error_does_not_propagate(self):
        from myelin_sdk.integrations.langchain.toolkit import MyelinToolkit

        tk = MyelinToolkit(api_key="test_key")
        mock_client = _mock_client()
        mock_client.finish = AsyncMock(side_effect=RuntimeError("boom"))
        tk._client = mock_client

        tk._state.session_id = "ses_fail"
        tk._state.active = True

        # Should not raise
        async with tk:
            pass

        mock_client.close.assert_awaited_once()


class TestHandlerSkipsToolTools:
    async def test_skips_memory_recall(self):
        from myelin_sdk.integrations.langchain.handler import MyelinCallbackHandler

        client = AsyncMock()
        handler = MyelinCallbackHandler(
            client=client, session_id="ses_test"
        )

        run_id = uuid4()
        await handler.on_tool_start(
            {"name": "memory_recall"},
            '{"task_description": "test"}',
            run_id=run_id,
        )

        assert run_id not in handler._pending_tools

    async def test_skips_memory_hint(self):
        from myelin_sdk.integrations.langchain.handler import MyelinCallbackHandler

        client = AsyncMock()
        handler = MyelinCallbackHandler(
            client=client, session_id="ses_test"
        )

        run_id = uuid4()
        await handler.on_tool_start(
            {"name": "memory_hint"},
            '{"step_number": 1}',
            run_id=run_id,
        )

        assert run_id not in handler._pending_tools

    async def test_skips_memory_finish(self):
        from myelin_sdk.integrations.langchain.handler import MyelinCallbackHandler

        client = AsyncMock()
        handler = MyelinCallbackHandler(
            client=client, session_id="ses_test"
        )

        run_id = uuid4()
        await handler.on_tool_start(
            {"name": "memory_finish"},
            '{}',
            run_id=run_id,
        )

        assert run_id not in handler._pending_tools

    async def test_does_not_skip_other_tools(self):
        from myelin_sdk.integrations.langchain.handler import MyelinCallbackHandler

        client = AsyncMock()
        handler = MyelinCallbackHandler(
            client=client, session_id="ses_test"
        )

        run_id = uuid4()
        await handler.on_tool_start(
            {"name": "search"},
            '{"q": "test"}',
            run_id=run_id,
        )

        assert run_id in handler._pending_tools


class TestHandlerDynamicSessionId:
    async def test_uses_state_session_id(self):
        from myelin_sdk.integrations.langchain.handler import MyelinCallbackHandler
        from myelin_sdk.integrations.langchain.state import _MyelinToolState

        client = AsyncMock()
        client.capture = AsyncMock()
        state = _MyelinToolState(session_id="ses_from_state")

        handler = MyelinCallbackHandler(
            client=client,
            session_id="",
            state=state,
        )

        run_id = uuid4()
        await handler.on_tool_start(
            {"name": "search"}, '{"q": "test"}', run_id=run_id
        )
        await handler.on_tool_end("results", run_id=run_id)

        call_kwargs = client.capture.call_args.kwargs
        assert call_kwargs["session_id"] == "ses_from_state"

    async def test_falls_back_to_static_session_id(self):
        from myelin_sdk.integrations.langchain.handler import MyelinCallbackHandler

        client = AsyncMock()
        client.capture = AsyncMock()

        handler = MyelinCallbackHandler(
            client=client,
            session_id="ses_static",
        )

        run_id = uuid4()
        await handler.on_tool_start(
            {"name": "search"}, '{"q": "test"}', run_id=run_id
        )
        await handler.on_tool_end("results", run_id=run_id)

        call_kwargs = client.capture.call_args.kwargs
        assert call_kwargs["session_id"] == "ses_static"

    async def test_skips_capture_when_no_session(self):
        from myelin_sdk.integrations.langchain.handler import MyelinCallbackHandler
        from myelin_sdk.integrations.langchain.state import _MyelinToolState

        client = AsyncMock()
        client.capture = AsyncMock()
        state = _MyelinToolState()  # no session_id yet

        handler = MyelinCallbackHandler(
            client=client,
            session_id="",
            state=state,
        )

        run_id = uuid4()
        await handler.on_tool_start(
            {"name": "search"}, '{"q": "test"}', run_id=run_id
        )
        await handler.on_tool_end("results", run_id=run_id)

        client.capture.assert_not_awaited()
