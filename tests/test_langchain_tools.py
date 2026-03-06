"""Tests for the LangChain autonomous tools (recall, hint, finish)."""

import json
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def _patch_langchain(patch_langchain):
    """Use the shared langchain patch from conftest."""


def _make_state():
    from myelin_sdk.integrations.langchain.state import _MyelinToolState

    return _MyelinToolState()


def _make_recall_tool(client=None, state=None, agent_id="default"):
    from myelin_sdk.integrations.langchain.tools import MemoryRecallTool

    if client is None:
        client = AsyncMock()
    if state is None:
        state = _make_state()
    return MemoryRecallTool(client=client, state=state, agent_id=agent_id), client, state


def _make_finish_tool(client=None, state=None):
    from myelin_sdk.integrations.langchain.tools import MemoryFinishTool

    if client is None:
        client = AsyncMock()
    if state is None:
        state = _make_state()
    return MemoryFinishTool(client=client, state=state), client, state


class TestMemoryRecallTool:
    async def test_hit_formats_content(self):
        from myelin_sdk.types import RecallResponse, WorkflowInfo

        client = AsyncMock()
        client.recall = AsyncMock(
            return_value=RecallResponse(
                session_id="ses_123",
                matched=True,
                workflow=WorkflowInfo(
                    id="wf_1",
                    description="Deploy app",
                    overview="Deploy the application to production.",
                    content="## Step 1\nBuild the Docker image\n\n## Step 2\nPush to registry",
                ),
            )
        )

        tool, _, state = _make_recall_tool(client=client)
        result = await tool._arun("deploy the app")

        assert "ses_123" in result
        assert "Deploy the application to production." in result
        assert "Build the Docker image" in result
        assert "Push to registry" in result
        assert "memory_finish" in result

        assert state.session_id == "ses_123"
        assert state.matched is True
        assert state.active is True
        assert state.workflow is not None

    async def test_miss_formats_freestyle(self):
        from myelin_sdk.types import RecallResponse

        client = AsyncMock()
        client.recall = AsyncMock(
            return_value=RecallResponse(
                session_id="ses_456",
                matched=False,
            )
        )

        tool, _, state = _make_recall_tool(client=client)
        result = await tool._arun("some new task")

        assert "ses_456" in result
        assert "No matching workflow found" in result
        assert "memory_finish" in result

        assert state.session_id == "ses_456"
        assert state.matched is False
        assert state.active is True

    async def test_error_returns_string(self):
        client = AsyncMock()
        client.recall = AsyncMock(side_effect=RuntimeError("network fail"))

        tool, _, state = _make_recall_tool(client=client)
        result = await tool._arun("test task")

        assert "Error:" in result
        assert "network fail" in result
        assert state.session_id is None
        assert state.active is False

    async def test_passes_agent_id(self):
        from myelin_sdk.types import RecallResponse

        client = AsyncMock()
        client.recall = AsyncMock(
            return_value=RecallResponse(session_id="s", matched=False)
        )

        tool, _, _ = _make_recall_tool(client=client, agent_id="my-agent")
        await tool._arun("task")

        client.recall.assert_awaited_once_with("task", "my-agent")


class TestMemoryFinishTool:
    async def test_returns_json(self):
        from myelin_sdk.types import FinishResponse

        client = AsyncMock()
        client.finish = AsyncMock(
            return_value=FinishResponse(
                session_id="ses_123",
                tool_calls_recorded=5,
                status="evaluation_queued",
                workflow_id="wf_1",
            )
        )

        state = _make_state()
        state.session_id = "ses_123"
        state.active = True

        tool, _, _ = _make_finish_tool(client=client, state=state)
        result = await tool._arun()
        parsed = json.loads(result)

        assert parsed["session_id"] == "ses_123"
        assert parsed["tool_calls_recorded"] == 5
        assert parsed["status"] == "evaluation_queued"
        assert parsed["workflow_id"] == "wf_1"
        assert state.active is False

    async def test_includes_warning(self):
        from myelin_sdk.types import FinishResponse

        client = AsyncMock()
        client.finish = AsyncMock(
            return_value=FinishResponse(
                session_id="ses_123",
                tool_calls_recorded=0,
                status="evaluation_queued",
                warning="No tool calls recorded",
            )
        )

        state = _make_state()
        state.session_id = "ses_123"
        state.active = True

        tool, _, _ = _make_finish_tool(client=client, state=state)
        result = await tool._arun()
        parsed = json.loads(result)

        assert parsed["warning"] == "No tool calls recorded"

    async def test_no_session_returns_error(self):
        tool, _, _ = _make_finish_tool()
        result = await tool._arun()

        assert "Error" in result
        assert "memory_recall" in result

    async def test_double_finish_returns_already_finished(self):
        state = _make_state()
        state.session_id = "ses_123"
        state.active = False

        tool, client, _ = _make_finish_tool(state=state)
        result = await tool._arun()

        assert "already finished" in result.lower()
        client.finish.assert_not_awaited()

    async def test_api_error_returns_string(self):
        client = AsyncMock()
        client.finish = AsyncMock(side_effect=RuntimeError("server error"))

        state = _make_state()
        state.session_id = "ses_123"
        state.active = True

        tool, _, _ = _make_finish_tool(client=client, state=state)
        result = await tool._arun()

        assert "Error:" in result
        assert "server error" in result
