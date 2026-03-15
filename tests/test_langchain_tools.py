"""Tests for the LangChain autonomous tools (search, start, finish)."""

import json
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def _patch_langchain(patch_langchain):
    """Use the shared langchain patch from conftest."""


def _make_state():
    from myelin_sdk.integrations.langchain.state import _MyelinToolState

    return _MyelinToolState()


def _make_search_tool(client=None, state=None):
    from myelin_sdk.integrations.langchain.tools import MemorySearchTool

    if client is None:
        client = AsyncMock()
    if state is None:
        state = _make_state()
    return MemorySearchTool(client=client, state=state), client, state


def _make_record_tool(client=None, state=None):
    from myelin_sdk.integrations.langchain.tools import MemoryRecordTool

    if client is None:
        client = AsyncMock()
    if state is None:
        state = _make_state()
    return MemoryRecordTool(client=client, state=state), client, state


def _make_finish_tool(client=None, state=None):
    from myelin_sdk.integrations.langchain.tools import MemoryFinishTool

    if client is None:
        client = AsyncMock()
    if state is None:
        state = _make_state()
    return MemoryFinishTool(client=client, state=state), client, state


class TestMemorySearchTool:
    async def test_hit_formats_content(self):
        from myelin_sdk.types import SearchMatch, SearchResult

        client = AsyncMock()
        client.search = AsyncMock(
            return_value=SearchResult(
                top_match=SearchMatch(
                    workflow_id="wf_1",
                    title="Deploy app",
                    description="Deploy the application to production.",
                    content="## Step 1\nBuild the Docker image\n\n## Step 2\nPush to registry",
                    score=0.95,
                ),
                other_matches=[
                    SearchMatch(
                        workflow_id="wf_2",
                        title="Rollback",
                        description="Rollback deployment",
                        score=0.72,
                    ),
                ],
            )
        )

        tool, _, state = _make_search_tool(client=client)
        result = await tool._arun("deploy the app")

        assert "wf_1" in result
        assert "Deploy app" in result
        assert "Build the Docker image" in result
        assert "Push to registry" in result
        assert "memory_record" in result
        assert "Rollback" in result
        assert state.last_search is not None

    async def test_miss_formats_freestyle(self):
        from myelin_sdk.types import SearchResult

        client = AsyncMock()
        client.search = AsyncMock(
            return_value=SearchResult(
                top_match=None,
                other_matches=[],
            )
        )

        tool, _, state = _make_search_tool(client=client)
        result = await tool._arun("some new task")

        assert "No matching workflows found" in result
        assert "memory_record" in result

    async def test_miss_with_other_matches(self):
        from myelin_sdk.types import SearchMatch, SearchResult

        client = AsyncMock()
        client.search = AsyncMock(
            return_value=SearchResult(
                top_match=None,
                other_matches=[
                    SearchMatch(
                        workflow_id="wf_1",
                        title="Deploy",
                        description="Deploy app",
                    ),
                ],
            )
        )

        tool, _, _ = _make_search_tool(client=client)
        result = await tool._arun("task")

        assert "No strong match" in result
        assert "Deploy" in result
        assert "memory_record" in result

    async def test_error_returns_string(self):
        client = AsyncMock()
        client.search = AsyncMock(side_effect=RuntimeError("network fail"))

        tool, _, state = _make_search_tool(client=client)
        result = await tool._arun("test task")

        assert "Error:" in result
        assert "network fail" in result


class TestMemoryRecordTool:
    async def test_start_with_workflow(self):
        from myelin_sdk.types import StartResult

        client = AsyncMock()
        client.start = AsyncMock(
            return_value=StartResult(
                session_id="ses_123",
                matched_workflow_id="wf_1",
            )
        )

        tool, _, state = _make_record_tool(client=client)
        result = await tool._arun(workflow_id="wf_1", task_description="deploy")

        assert "ses_123" in result
        assert "wf_1" in result
        assert "memory_finish" in result
        assert state.session_id == "ses_123"
        assert state.matched_workflow_id == "wf_1"
        assert state.active is True

    async def test_start_without_workflow(self):
        from myelin_sdk.types import StartResult

        client = AsyncMock()
        client.start = AsyncMock(
            return_value=StartResult(
                session_id="ses_456",
                matched_workflow_id=None,
            )
        )

        tool, _, state = _make_record_tool(client=client)
        result = await tool._arun(task_description="new task")

        assert "ses_456" in result
        assert "memory_finish" in result
        assert state.session_id == "ses_456"
        assert state.active is True

    async def test_error_returns_string(self):
        client = AsyncMock()
        client.start = AsyncMock(side_effect=RuntimeError("network fail"))

        tool, _, state = _make_record_tool(client=client)
        result = await tool._arun()

        assert "Error:" in result
        assert "network fail" in result
        assert state.session_id is None
        assert state.active is False


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
        assert "memory_record" in result

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
