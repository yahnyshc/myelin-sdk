"""Tests for the LangChain callback handler."""

import sys
import types
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest


class FakeGeneration:
    def __init__(self, text: str):
        self.text = text


class FakeLLMResult:
    def __init__(self, texts: list[str]):
        self.generations = [[FakeGeneration(t) for t in texts]]


class _AsyncCallbackHandler:
    pass


@pytest.fixture(autouse=True)
def _patch_langchain(monkeypatch):
    """Patch langchain_core imports so tests work without langchain installed."""
    langchain_core = types.ModuleType("langchain_core")
    callbacks_mod = types.ModuleType("langchain_core.callbacks")
    outputs_mod = types.ModuleType("langchain_core.outputs")

    callbacks_mod.AsyncCallbackHandler = _AsyncCallbackHandler
    outputs_mod.LLMResult = FakeLLMResult

    langchain_core.callbacks = callbacks_mod
    langchain_core.outputs = outputs_mod

    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", callbacks_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.outputs", outputs_mod)

    # Force reimport of handler module so it picks up our mocks
    for mod_name in list(sys.modules):
        if "myelin_sdk.langchain" in mod_name:
            monkeypatch.delitem(sys.modules, mod_name)


def _make_handler(client=None, session_id="ses_test"):
    from myelin_sdk.langchain.handler import MyelinCallbackHandler

    if client is None:
        client = AsyncMock()
        client.capture = AsyncMock()
    return MyelinCallbackHandler(client=client, session_id=session_id), client


class TestOnLlmEnd:
    async def test_buffers_reasoning(self):
        handler, _ = _make_handler()
        run_id = uuid4()
        result = FakeLLMResult(["I should read the file first"])
        await handler.on_llm_end(result, run_id=run_id)
        assert run_id in handler._reasoning
        assert handler._reasoning[run_id] == "I should read the file first"

    async def test_multiple_generations(self):
        handler, _ = _make_handler()
        run_id = uuid4()
        result = FakeLLMResult(["part one", "part two"])
        await handler.on_llm_end(result, run_id=run_id)
        assert "part one" in handler._reasoning[run_id]
        assert "part two" in handler._reasoning[run_id]

    async def test_empty_text_ignored(self):
        handler, _ = _make_handler()
        run_id = uuid4()
        result = FakeLLMResult([""])
        await handler.on_llm_end(result, run_id=run_id)
        assert run_id not in handler._reasoning

    async def test_exception_does_not_raise(self):
        handler, _ = _make_handler()
        run_id = uuid4()
        bad_result = MagicMock()
        bad_result.generations = None
        await handler.on_llm_end(bad_result, run_id=run_id)


class TestOnToolStart:
    async def test_records_pending_tool(self):
        handler, _ = _make_handler()
        run_id = uuid4()
        parent_id = uuid4()
        await handler.on_tool_start(
            {"name": "search"},
            '{"query": "python"}',
            run_id=run_id,
            parent_run_id=parent_id,
        )
        assert run_id in handler._pending_tools
        pending = handler._pending_tools[run_id]
        assert pending["name"] == "search"
        assert pending["input"] == {"query": "python"}
        assert pending["parent_run_id"] == parent_id

    async def test_non_json_input(self):
        handler, _ = _make_handler()
        run_id = uuid4()
        await handler.on_tool_start(
            {"name": "bash"},
            "just a string",
            run_id=run_id,
        )
        assert handler._pending_tools[run_id]["input"] == {"input": "just a string"}

    async def test_missing_name(self):
        handler, _ = _make_handler()
        run_id = uuid4()
        await handler.on_tool_start({}, '{}', run_id=run_id)
        assert handler._pending_tools[run_id]["name"] == "unknown"


class TestOnToolEnd:
    async def test_captures_tool_call(self):
        handler, client = _make_handler()
        llm_run_id = uuid4()
        tool_run_id = uuid4()

        await handler.on_llm_end(
            FakeLLMResult(["Let me search for that"]),
            run_id=llm_run_id,
        )
        await handler.on_tool_start(
            {"name": "search"},
            '{"q": "test"}',
            run_id=tool_run_id,
            parent_run_id=llm_run_id,
        )
        await handler.on_tool_end("result: found it", run_id=tool_run_id)

        client.capture.assert_awaited_once()
        call_kwargs = client.capture.call_args.kwargs
        assert call_kwargs["session_id"] == "ses_test"
        assert call_kwargs["tool_name"] == "search"
        assert call_kwargs["tool_input"] == {"q": "test"}
        assert call_kwargs["tool_response"] == "result: found it"
        assert call_kwargs["reasoning"] == "Let me search for that"

    async def test_no_reasoning_when_no_parent(self):
        handler, client = _make_handler()
        tool_run_id = uuid4()
        await handler.on_tool_start(
            {"name": "bash"},
            '{"cmd": "ls"}',
            run_id=tool_run_id,
            parent_run_id=None,
        )
        await handler.on_tool_end("files", run_id=tool_run_id)

        client.capture.assert_awaited_once()
        assert client.capture.call_args.kwargs["reasoning"] is None

    async def test_reasoning_consumed_once(self):
        handler, client = _make_handler()
        llm_run_id = uuid4()
        tool1 = uuid4()
        tool2 = uuid4()

        await handler.on_llm_end(
            FakeLLMResult(["thinking"]), run_id=llm_run_id
        )
        await handler.on_tool_start(
            {"name": "t1"}, '{}', run_id=tool1, parent_run_id=llm_run_id
        )
        await handler.on_tool_start(
            {"name": "t2"}, '{}', run_id=tool2, parent_run_id=llm_run_id
        )

        await handler.on_tool_end("r1", run_id=tool1)
        await handler.on_tool_end("r2", run_id=tool2)

        calls = client.capture.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["reasoning"] == "thinking"
        assert calls[1].kwargs["reasoning"] is None

    async def test_unknown_run_id_ignored(self):
        handler, client = _make_handler()
        await handler.on_tool_end("output", run_id=uuid4())
        client.capture.assert_not_awaited()

    async def test_truncates_long_output(self):
        handler, client = _make_handler()
        tool_run_id = uuid4()
        await handler.on_tool_start(
            {"name": "read"}, '{}', run_id=tool_run_id
        )
        long_output = "x" * 20000
        await handler.on_tool_end(long_output, run_id=tool_run_id)

        captured = client.capture.call_args.kwargs["tool_response"]
        assert len(captured) < 20000
        assert "20000 chars" in captured


class TestOnToolError:
    async def test_captures_error(self):
        handler, client = _make_handler()
        tool_run_id = uuid4()
        await handler.on_tool_start(
            {"name": "bash"}, '{}', run_id=tool_run_id
        )
        await handler.on_tool_error(
            RuntimeError("command failed"), run_id=tool_run_id
        )

        client.capture.assert_awaited_once()
        resp = client.capture.call_args.kwargs["tool_response"]
        assert "ERROR: command failed" in resp


class TestFireAndForget:
    async def test_capture_failure_does_not_raise(self):
        handler, client = _make_handler()
        client.capture.side_effect = Exception("network error")

        tool_run_id = uuid4()
        await handler.on_tool_start(
            {"name": "bash"}, '{}', run_id=tool_run_id
        )
        await handler.on_tool_end("output", run_id=tool_run_id)
