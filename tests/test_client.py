"""Tests for MyelinClient."""

import httpx
import pytest

from myelin_sdk.client import MyelinClient


def _make_transport(responses: list[dict]):
    """Create a mock transport returning canned responses in order."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        resp = responses[idx]
        return httpx.Response(
            status_code=resp.get("status", 200),
            json=resp.get("json", {}),
        )

    return httpx.MockTransport(handler)


def _client_with(responses: list[dict]) -> MyelinClient:
    client = MyelinClient(api_key="test-key", base_url="https://test.myelin.dev")
    client._http = httpx.AsyncClient(
        transport=_make_transport(responses),
        base_url="https://test.myelin.dev",
        headers={"Authorization": "Bearer test-key"},
    )
    return client


class TestRecall:
    async def test_recall_hit(self):
        client = _client_with([{
            "json": {
                "session_id": "ses_123",
                "matched": True,
                "workflow": {
                    "id": "wf_1",
                    "description": "test workflow",
                    "total_steps": 3,
                    "overview": "overview text",
                    "skeleton": "1. step one\n2. step two\n3. step three",
                },
            },
        }])
        resp = await client.recall("do something")
        assert resp.session_id == "ses_123"
        assert resp.matched is True
        assert resp.workflow is not None
        assert resp.workflow.id == "wf_1"
        assert resp.workflow.total_steps == 3
        await client.close()

    async def test_recall_miss(self):
        client = _client_with([{
            "json": {"session_id": "ses_456", "matched": False},
        }])
        resp = await client.recall("unknown task")
        assert resp.session_id == "ses_456"
        assert resp.matched is False
        assert resp.workflow is None
        await client.close()

    async def test_recall_error(self):
        client = _client_with([{"status": 500, "json": {"error": "internal"}}])
        with pytest.raises(httpx.HTTPStatusError):
            await client.recall("fail")
        await client.close()


class TestCapture:
    async def test_capture_basic(self):
        client = _client_with([{"json": {"status": "ok"}}])
        resp = await client.capture(
            session_id="ses_1",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_response="file1\nfile2",
        )
        assert resp.status == "ok"
        await client.close()

    async def test_capture_with_reasoning(self):
        client = _client_with([{"json": {"status": "ok"}}])
        resp = await client.capture(
            session_id="ses_1",
            tool_name="Read",
            tool_input={"path": "/file"},
            tool_response="contents",
            reasoning="I need to read this file",
            client_ts=1000.0,
        )
        assert resp.status == "ok"
        await client.close()


class TestFinish:
    async def test_finish(self):
        client = _client_with([{
            "json": {
                "session_id": "ses_1",
                "tool_calls_recorded": 5,
                "status": "evaluated",
                "workflow_id": "wf_new",
            },
        }])
        resp = await client.finish("ses_1")
        assert resp.session_id == "ses_1"
        assert resp.tool_calls_recorded == 5
        assert resp.status == "evaluated"
        assert resp.workflow_id == "wf_new"
        await client.close()


class TestHint:
    async def test_hint(self):
        client = _client_with([{
            "json": {
                "session_id": "ses_1",
                "step_number": 2,
                "detail": "Run the migration script",
            },
        }])
        resp = await client.hint("ses_1", 2)
        assert resp.session_id == "ses_1"
        assert resp.step_number == 2
        assert resp.detail == "Run the migration script"
        await client.close()


class TestHints:
    async def test_hints(self):
        client = _client_with([{
            "json": {
                "session_id": "ses_1",
                "hints": {"1": "Do first thing", "2": "Do second thing"},
            },
        }])
        resp = await client.hints("ses_1")
        assert resp.session_id == "ses_1"
        assert resp.hints == {1: "Do first thing", 2: "Do second thing"}
        await client.close()


class TestRedaction:
    async def test_default_redaction_applied(self):
        """Client applies redaction to tool_input and tool_response by default."""
        requests_sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json
            requests_sent.append(_json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})

        client = MyelinClient(api_key="test-key", base_url="https://test.myelin.dev")
        client._http = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://test.myelin.dev",
            headers={"Authorization": "Bearer test-key"},
        )
        await client.capture(
            session_id="ses_1",
            tool_name="Bash",
            tool_input={"api_key": "sk-ant-api03-secretsecretsecretsecret"},
            tool_response="token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn",
        )
        body = requests_sent[0]
        assert body["tool_input"]["api_key"] == "[REDACTED]"
        assert "ghp_" not in body["tool_response"]
        await client.close()

    async def test_redaction_disabled(self):
        """Redaction can be disabled via config."""
        from myelin_sdk.redact import RedactionConfig

        requests_sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json
            requests_sent.append(_json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})

        client = MyelinClient(
            api_key="test-key",
            base_url="https://test.myelin.dev",
            redaction=RedactionConfig(enabled=False),
        )
        client._http = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://test.myelin.dev",
            headers={"Authorization": "Bearer test-key"},
        )
        secret = "sk-ant-api03-secretsecretsecretsecret"
        await client.capture(
            session_id="ses_1",
            tool_name="Bash",
            tool_input={"key": secret},
            tool_response=secret,
        )
        body = requests_sent[0]
        assert body["tool_input"]["key"] == secret
        assert body["tool_response"] == secret
        await client.close()

    async def test_reasoning_redacted(self):
        """Reasoning containing secrets is redacted."""
        requests_sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json
            requests_sent.append(_json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})

        client = MyelinClient(api_key="test-key", base_url="https://test.myelin.dev")
        client._http = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://test.myelin.dev",
            headers={"Authorization": "Bearer test-key"},
        )
        await client.capture(
            session_id="ses_1",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_response="ok",
            reasoning="Use key ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn",
        )
        body = requests_sent[0]
        assert "ghp_" not in body["reasoning"]
        assert "[REDACTED]" in body["reasoning"]
        await client.close()


class TestContextManager:
    async def test_async_context_manager(self):
        client = _client_with([{"json": {"session_id": "ses_1", "matched": False}}])
        async with client:
            resp = await client.recall("task")
            assert resp.session_id == "ses_1"
        # Client should be closed after exiting context
