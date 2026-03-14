"""Tests for MyelinClient."""

import httpx
import pytest

from myelin_sdk._utils import validate_base_url
from myelin_sdk.client import MyelinClient
from myelin_sdk.errors import MyelinAPIError


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


class TestSearch:
    async def test_search_with_match(self):
        client = _client_with([{
            "json": {
                "top_match": {
                    "workflow_id": "wf_1",
                    "title": "Deploy app",
                    "description": "Deploy to production",
                    "content": "## Step 1\nBuild\n\n## Step 2\nDeploy",
                    "score": 0.95,
                },
                "other_matches": [
                    {
                        "workflow_id": "wf_2",
                        "title": "Rollback",
                        "description": "Rollback deployment",
                        "score": 0.72,
                    },
                ],
            },
        }])
        resp = await client.search("deploy the app")
        assert resp.top_match is not None
        assert resp.top_match.workflow_id == "wf_1"
        assert resp.top_match.score == 0.95
        assert resp.top_match.content is not None
        assert len(resp.other_matches) == 1
        assert resp.other_matches[0].workflow_id == "wf_2"
        await client.close()

    async def test_search_no_match(self):
        client = _client_with([{
            "json": {"top_match": None, "other_matches": []},
        }])
        resp = await client.search("unknown task")
        assert resp.top_match is None
        assert resp.other_matches == []
        await client.close()

    async def test_search_list_all(self):
        client = _client_with([{
            "json": {
                "top_match": None,
                "other_matches": [
                    {
                        "workflow_id": "wf_1",
                        "title": "Deploy",
                        "description": "Deploy app",
                    },
                    {
                        "workflow_id": "wf_2",
                        "title": "Rollback",
                        "description": "Rollback deployment",
                    },
                ],
            },
        }])
        resp = await client.search()
        assert resp.top_match is None
        assert len(resp.other_matches) == 2
        await client.close()

    async def test_search_error(self):
        client = _client_with([{"status": 500, "json": {"error": "internal"}}])
        with pytest.raises(httpx.HTTPStatusError):
            await client.search("fail")
        await client.close()


class TestStart:
    async def test_start_with_workflow(self):
        client = _client_with([{
            "json": {
                "session_id": "ses_123",
                "matched_workflow_id": "wf_1",
            },
        }])
        resp = await client.start(workflow_id="wf_1", task_description="deploy")
        assert resp.session_id == "ses_123"
        assert resp.matched_workflow_id == "wf_1"
        await client.close()

    async def test_start_without_workflow(self):
        client = _client_with([{
            "json": {"session_id": "ses_456", "matched_workflow_id": None},
        }])
        resp = await client.start(task_description="new task")
        assert resp.session_id == "ses_456"
        assert resp.matched_workflow_id is None
        await client.close()

    async def test_start_error(self):
        client = _client_with([{"status": 500, "json": {"error": "internal"}}])
        with pytest.raises(httpx.HTTPStatusError):
            await client.start()
        await client.close()

    async def test_start_error_is_myelin_api_error(self):
        client = _client_with([{"status": 500, "json": {"error": "internal"}}])
        with pytest.raises(MyelinAPIError):
            await client.start()
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


class TestBaseUrlValidation:
    def test_https_allowed(self):
        validate_base_url("https://myelin.fly.dev")

    def test_https_custom_domain(self):
        validate_base_url("https://myelin.internal.company.com:8443")

    def test_http_localhost_allowed(self):
        validate_base_url("http://localhost:8080")

    def test_http_127_allowed(self):
        validate_base_url("http://127.0.0.1:19876")

    def test_http_ipv6_loopback_allowed(self):
        validate_base_url("http://[::1]:8080")

    def test_http_remote_rejected(self):
        with pytest.raises(ValueError, match="HTTPS is required"):
            validate_base_url("http://evil.com")

    def test_http_remote_with_port_rejected(self):
        with pytest.raises(ValueError, match="HTTPS is required"):
            validate_base_url("http://attacker.com:443")

    def test_ftp_rejected(self):
        with pytest.raises(ValueError, match="Invalid base_url scheme"):
            validate_base_url("ftp://files.example.com")

    def test_no_scheme_rejected(self):
        with pytest.raises(ValueError, match="Invalid base_url scheme"):
            validate_base_url("myelin.fly.dev")

    def test_client_rejects_http_remote(self):
        with pytest.raises(ValueError, match="HTTPS is required"):
            MyelinClient(api_key="test-key", base_url="http://evil.com")

    def test_client_accepts_https(self):
        client = MyelinClient(api_key="test-key", base_url="https://myelin.fly.dev")
        assert client is not None

    def test_client_accepts_http_localhost(self):
        client = MyelinClient(api_key="test-key", base_url="http://localhost:8080")
        assert client is not None


class TestContextManager:
    async def test_async_context_manager(self):
        client = _client_with([{
            "json": {"session_id": "ses_1", "matched_workflow_id": None},
        }])
        async with client:
            resp = await client.start(task_description="task")
            assert resp.session_id == "ses_1"
        # Client should be closed after exiting context


class TestMyelinAPIError:
    async def test_401_includes_hint(self):
        client = _client_with([{"status": 401, "json": {"error": "Unauthorized"}}])
        with pytest.raises(MyelinAPIError, match="Check your API key"):
            await client.search("task")
        await client.close()

    async def test_404_includes_hint(self):
        client = _client_with([{"status": 404, "json": {"error": "Not found"}}])
        with pytest.raises(MyelinAPIError, match="may have expired"):
            await client.finish("ses_gone")
        await client.close()

    async def test_429_includes_hint(self):
        client = _client_with([{"status": 429, "json": {"error": "Too many requests"}}])
        with pytest.raises(MyelinAPIError, match="Rate limit"):
            await client.search("task")
        await client.close()

    async def test_server_error_message_in_detail(self):
        client = _client_with([{"status": 500, "json": {"error": "DB connection lost"}}])
        with pytest.raises(MyelinAPIError, match="DB connection lost"):
            await client.search("task")
        await client.close()

    async def test_caught_by_httpx_handler(self):
        """MyelinAPIError is still caught by except httpx.HTTPStatusError."""
        client = _client_with([{"status": 401, "json": {"error": "Unauthorized"}}])
        with pytest.raises(httpx.HTTPStatusError):
            await client.search("task")
        await client.close()

    async def test_error_format(self):
        client = _client_with([{"status": 401, "json": {"error": "Unauthorized"}}])
        try:
            await client.search("task")
        except MyelinAPIError as e:
            assert str(e).startswith("Myelin API error (401)")
            assert "Unauthorized" in str(e)
            assert e.status_code == 401
        await client.close()
