"""Tests for the capture.py PostToolUse hook (SDK copy).

Runs the hook as a subprocess to test it exactly as Claude Code would.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pytest

HOOK_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "myelin_sdk"
    / "claude_code"
    / "capture.py"
)

_COMMON_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "MYELIN_API_KEY": "test-key",
}


@pytest.fixture
def project_dir():
    """Create a temporary directory to act as CLAUDE_PROJECT_DIR."""
    with tempfile.TemporaryDirectory() as d:
        yield d


_DEFAULT_MYELIN_URL = "http://127.0.0.1:19876"


def _base_env(project_dir: str) -> dict:
    return {**_COMMON_ENV, "MYELIN_URL": _DEFAULT_MYELIN_URL, "CLAUDE_PROJECT_DIR": project_dir}


def run_hook(
    stdin_data: dict,
    project_dir: str,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run capture.py as a subprocess with the given stdin JSON and env."""
    final_env = {**_base_env(project_dir), **(env or {})}
    return subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=json.dumps(stdin_data),
        capture_output=True,
        text=True,
        env=final_env,
        timeout=10,
    )


def session_file(project_dir: str, cc_session_id: str) -> Path:
    import re
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", cc_session_id)
    return Path(project_dir) / ".claude" / ".myelin-sessions" / safe_id


@pytest.fixture
def cc_session_id():
    """Provide a unique CC session ID."""
    return f"test_{os.getpid()}_{id(object())}"


class TestMissingFields:
    def test_missing_tool_name(self, cc_session_id, project_dir):
        result = run_hook({"session_id": cc_session_id}, project_dir)
        assert result.returncode == 0

    def test_missing_session_id(self, project_dir):
        result = run_hook({"tool_name": "Bash"}, project_dir)
        assert result.returncode == 0

    def test_empty_stdin(self, project_dir):
        result = subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            input="",
            capture_output=True,
            text=True,
            env=_base_env(project_dir),
            timeout=10,
        )
        assert result.returncode == 0

    def test_invalid_json(self, project_dir):
        result = subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            input="not json",
            capture_output=True,
            text=True,
            env=_base_env(project_dir),
            timeout=10,
        )
        assert result.returncode == 0


class TestRecall:
    def test_creates_session_file(self, cc_session_id, project_dir):
        result = run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": cc_session_id,
            "tool_response": {"session_id": "ses_abc123"},
        }, project_dir)
        assert result.returncode == 0
        sf = session_file(project_dir, cc_session_id)
        assert sf.exists()
        assert sf.read_text() == "ses_abc123"

    def test_nested_result_string(self, cc_session_id, project_dir):
        """Handle {result: '{"session_id": "..."}'}."""
        result = run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": cc_session_id,
            "tool_response": {"result": json.dumps({"session_id": "ses_nested"})},
        }, project_dir)
        assert result.returncode == 0
        assert session_file(project_dir, cc_session_id).read_text() == "ses_nested"

    def test_nested_result_object(self, cc_session_id, project_dir):
        """Handle {result: {session_id: "..."}}."""
        result = run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": cc_session_id,
            "tool_response": {"result": {"session_id": "ses_obj"}},
        }, project_dir)
        assert result.returncode == 0
        assert session_file(project_dir, cc_session_id).read_text() == "ses_obj"

    def test_string_response(self, cc_session_id, project_dir):
        """Handle tool_response as a raw JSON string."""
        result = run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": cc_session_id,
            "tool_response": json.dumps({"session_id": "ses_str"}),
        }, project_dir)
        assert result.returncode == 0
        assert session_file(project_dir, cc_session_id).read_text() == "ses_str"

    def test_missing_env_vars(self, cc_session_id, project_dir):
        result = run_hook(
            {
                "tool_name": "mcp__myelin__memory_recall",
                "session_id": cc_session_id,
                "tool_response": {"session_id": "ses_x"},
            },
            project_dir,
            env={"MYELIN_URL": "", "MYELIN_API_KEY": ""},
        )
        assert result.returncode == 2
        assert "MYELIN_URL" in result.stderr

    def test_no_session_id_in_response(self, cc_session_id, project_dir):
        """If tool_response has no session_id, don't create file."""
        result = run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": cc_session_id,
            "tool_response": {"matches": []},
        }, project_dir)
        assert result.returncode == 0
        assert not session_file(project_dir, cc_session_id).exists()

    def test_no_project_dir_skips_session_file(self, cc_session_id):
        """Without CLAUDE_PROJECT_DIR, recall succeeds but no file is written."""
        env = {**_COMMON_ENV, "MYELIN_URL": _DEFAULT_MYELIN_URL, "CLAUDE_PROJECT_DIR": ""}
        result = subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            input=json.dumps({
                "tool_name": "mcp__myelin__memory_recall",
                "session_id": cc_session_id,
                "tool_response": {"session_id": "ses_noproj"},
            }),
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert result.returncode == 0


class TestFinish:
    def test_deletes_session_file(self, cc_session_id, project_dir):
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_todelete")

        result = run_hook({
            "tool_name": "mcp__myelin__memory_finish",
            "session_id": cc_session_id,
        }, project_dir)
        assert result.returncode == 0
        assert not sf.exists()

    def test_missing_file_ok(self, cc_session_id, project_dir):
        result = run_hook({
            "tool_name": "mcp__myelin__memory_finish",
            "session_id": cc_session_id,
        }, project_dir)
        assert result.returncode == 0


class TestSkipMyelinTools:
    def test_skips_own_tools(self, cc_session_id, project_dir):
        result = run_hook({
            "tool_name": "mcp__myelin__some_other_tool",
            "session_id": cc_session_id,
        }, project_dir)
        assert result.returncode == 0


class TestNoActiveSession:
    def test_no_session_file_skips(self, cc_session_id, project_dir):
        session_file(project_dir, cc_session_id).unlink(missing_ok=True)
        result = run_hook({
            "tool_name": "Bash",
            "session_id": cc_session_id,
            "tool_input": {"command": "ls"},
            "tool_response": "file1\nfile2",
        }, project_dir)
        assert result.returncode == 0


class CaptureHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that records POST bodies."""

    captured = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        CaptureHandler.captured.append(body)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, format, *args):
        pass  # suppress logs


@pytest.fixture
def capture_server():
    """Start a local HTTP server on a random free port to receive capture requests."""
    CaptureHandler.captured = []
    server = HTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


def _capture_env(capture_server) -> dict:
    """Return env override pointing MYELIN_URL at the capture server."""
    port = capture_server.server_address[1]
    return {"MYELIN_URL": f"http://127.0.0.1:{port}"}


class TestCapture:
    def test_posts_to_server(self, cc_session_id, project_dir, capture_server):
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_capture")

        result = run_hook({
            "tool_name": "Bash",
            "session_id": cc_session_id,
            "tool_input": {"command": "echo hello"},
            "tool_response": "hello",
        }, project_dir, env=_capture_env(capture_server))
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        body = CaptureHandler.captured[0]
        assert body["session_id"] == "ses_capture"
        assert body["tool_name"] == "Bash"
        assert body["tool_input"] == {"command": "echo hello"}
        assert body["tool_response"] == "hello"
        assert isinstance(body["client_ts"], float)
        assert body["client_ts"] > 0

    def test_truncates_long_response(self, cc_session_id, project_dir, capture_server):
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_trunc")

        long_response = "x" * 20000
        result = run_hook({
            "tool_name": "Write",
            "session_id": cc_session_id,
            "tool_input": {"path": "/file"},
            "tool_response": long_response,
        }, project_dir, env=_capture_env(capture_server))
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        resp = CaptureHandler.captured[0]["tool_response"]
        assert resp.startswith("x" * 4000)
        assert resp.endswith("x" * 4000)
        assert "20000 chars" in resp

    def test_captures_reasoning_from_transcript(self, cc_session_id, project_dir, capture_server):
        """Reasoning is extracted from the transcript and sent in the payload."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_reasoning")

        transcript = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        transcript.write(json.dumps({
            "message": {
                "id": "msg_r1",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I need to check the config."},
                    {"type": "text", "text": "Let me read the config file."},
                    {"type": "tool_use", "id": "toolu_reasoning1", "name": "Read"},
                ],
            }
        }) + "\n")
        transcript.flush()

        try:
            result = run_hook({
                "tool_name": "Write",
                "session_id": cc_session_id,
                "tool_use_id": "toolu_reasoning1",
                "transcript_path": transcript.name,
                "tool_input": {"path": "/etc/config"},
                "tool_response": "key=value",
            }, project_dir, env=_capture_env(capture_server))
            assert result.returncode == 0
            assert len(CaptureHandler.captured) == 1
            body = CaptureHandler.captured[0]
            assert "reasoning" in body
            assert "I need to check the config" in body["reasoning"]
            assert "Let me read the config file" in body["reasoning"]
        finally:
            os.unlink(transcript.name)

    def test_no_reasoning_without_transcript(self, cc_session_id, project_dir, capture_server):
        """No reasoning field when transcript_path is missing."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_noreason")

        result = run_hook({
            "tool_name": "Bash",
            "session_id": cc_session_id,
            "tool_use_id": "toolu_noreason",
            "tool_input": {"command": "ls"},
            "tool_response": "file1",
        }, project_dir, env=_capture_env(capture_server))
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        assert "reasoning" not in CaptureHandler.captured[0]

    def test_http_error_exits_zero(self, cc_session_id, project_dir):
        """Network failure (no server) should not block the agent."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_fail")

        result = run_hook(
            {
                "tool_name": "Bash",
                "session_id": cc_session_id,
                "tool_input": {},
                "tool_response": "out",
            },
            project_dir,
            env={"MYELIN_URL": "http://127.0.0.1:19877", "MYELIN_API_KEY": "k"},
        )
        assert result.returncode == 0
        assert "capture failed" in result.stderr


class TestInvestigationTools:
    """Investigation tools (Read, Glob, Grep) are captured input-only."""

    def test_read_captured_with_empty_response(self, cc_session_id, project_dir, capture_server):
        """Read tool is captured but tool_response is stripped."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_inv1")

        result = run_hook({
            "tool_name": "Read",
            "session_id": cc_session_id,
            "tool_input": {"file_path": "/src/main.py"},
            "tool_response": "def main():\n    print('hello world')\n" * 100,
        }, project_dir, env=_capture_env(capture_server))
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        body = CaptureHandler.captured[0]
        assert body["tool_name"] == "Read"
        assert body["tool_input"] == {"file_path": "/src/main.py"}
        assert body["tool_response"] == ""
        assert body["session_id"] == "ses_inv1"

    def test_glob_captured_with_empty_response(self, cc_session_id, project_dir, capture_server):
        """Glob tool is captured but tool_response is stripped."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_inv2")

        result = run_hook({
            "tool_name": "Glob",
            "session_id": cc_session_id,
            "tool_input": {"pattern": "**/*.py"},
            "tool_response": "src/a.py\nsrc/b.py\nsrc/c.py",
        }, project_dir, env=_capture_env(capture_server))
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        body = CaptureHandler.captured[0]
        assert body["tool_name"] == "Glob"
        assert body["tool_input"] == {"pattern": "**/*.py"}
        assert body["tool_response"] == ""

    def test_grep_captured_with_empty_response(self, cc_session_id, project_dir, capture_server):
        """Grep tool is captured but tool_response is stripped."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_inv3")

        result = run_hook({
            "tool_name": "Grep",
            "session_id": cc_session_id,
            "tool_input": {"pattern": "def main", "path": "/src"},
            "tool_response": "src/main.py:1:def main():",
        }, project_dir, env=_capture_env(capture_server))
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        body = CaptureHandler.captured[0]
        assert body["tool_name"] == "Grep"
        assert body["tool_input"] == {"pattern": "def main", "path": "/src"}
        assert body["tool_response"] == ""

    def test_action_tool_still_has_response(self, cc_session_id, project_dir, capture_server):
        """Non-investigation tools (Bash) still capture full response."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_inv4")

        result = run_hook({
            "tool_name": "Bash",
            "session_id": cc_session_id,
            "tool_input": {"command": "echo hello"},
            "tool_response": "hello",
        }, project_dir, env=_capture_env(capture_server))
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        body = CaptureHandler.captured[0]
        assert body["tool_response"] == "hello"

    def test_investigation_tool_reasoning_still_captured(
        self, cc_session_id, project_dir, capture_server
    ):
        """Reasoning is still captured for investigation tools."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_inv5")

        transcript = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        transcript.write(json.dumps({
            "message": {
                "id": "msg_inv",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I need to check the config."},
                    {"type": "tool_use", "id": "toolu_inv1", "name": "Read"},
                ],
            }
        }) + "\n")
        transcript.flush()

        try:
            result = run_hook({
                "tool_name": "Read",
                "session_id": cc_session_id,
                "tool_use_id": "toolu_inv1",
                "transcript_path": transcript.name,
                "tool_input": {"file_path": "/etc/config"},
                "tool_response": "key=value\nsecret=hidden",
            }, project_dir, env=_capture_env(capture_server))
            assert result.returncode == 0
            assert len(CaptureHandler.captured) == 1
            body = CaptureHandler.captured[0]
            assert body["tool_response"] == ""
            assert "reasoning" in body
            assert "I need to check the config" in body["reasoning"]
        finally:
            os.unlink(transcript.name)


class TestToolFailure:
    """PostToolUseFailure sends 'error' without 'tool_response'."""

    def test_error_captured_with_is_error_flag(self, cc_session_id, project_dir, capture_server):
        """Failure payload includes is_error=True and the error message."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_fail1")

        result = run_hook({
            "tool_name": "Bash",
            "session_id": cc_session_id,
            "tool_input": {"command": "false"},
            "error": "Command exited with code 1",
        }, project_dir, env=_capture_env(capture_server))
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        body = CaptureHandler.captured[0]
        assert body["is_error"] is True
        assert body["tool_response"] == "Command exited with code 1"
        assert body["tool_name"] == "Bash"

    def test_investigation_tool_error_preserves_message(
        self, cc_session_id, project_dir, capture_server
    ):
        """Read/Glob/Grep failures capture the error (not stripped like success)."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_fail2")

        result = run_hook({
            "tool_name": "Read",
            "session_id": cc_session_id,
            "tool_input": {"file_path": "/nonexistent"},
            "error": "File not found: /nonexistent",
        }, project_dir, env=_capture_env(capture_server))
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        body = CaptureHandler.captured[0]
        assert body["is_error"] is True
        assert body["tool_response"] == "File not found: /nonexistent"
        assert body["tool_name"] == "Read"

    def test_success_has_no_is_error(self, cc_session_id, project_dir, capture_server):
        """Normal success payloads should not have is_error field."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_fail3")

        result = run_hook({
            "tool_name": "Bash",
            "session_id": cc_session_id,
            "tool_input": {"command": "echo ok"},
            "tool_response": "ok",
        }, project_dir, env=_capture_env(capture_server))
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        assert "is_error" not in CaptureHandler.captured[0]

    def test_recall_failure_no_session_file(self, cc_session_id, project_dir):
        """Recall failure (error, no tool_response) should not create session file."""
        result = run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": cc_session_id,
            "error": "Connection refused",
        }, project_dir)
        assert result.returncode == 0
        assert not session_file(project_dir, cc_session_id).exists()

    def test_finish_failure_still_cleans_up(self, cc_session_id, project_dir):
        """Finish failure still removes session file."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_cleanup")

        result = run_hook({
            "tool_name": "mcp__myelin__memory_finish",
            "session_id": cc_session_id,
            "error": "Server error",
        }, project_dir)
        assert result.returncode == 0
        assert not sf.exists()

    def test_skipped_tool_failure_still_skipped(self, cc_session_id, project_dir, capture_server):
        """Failures for skipped tools (TaskCreate etc.) are still skipped."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_skip")

        result = run_hook({
            "tool_name": "TaskCreate",
            "session_id": cc_session_id,
            "error": "some error",
        }, project_dir, env=_capture_env(capture_server))
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 0


class TestRedaction:
    """Test redaction in the subprocess capture hook."""

    @pytest.fixture
    def redaction_config(self):
        """Create a temporary redaction.json from default config."""
        from myelin_sdk.redact import build_default_redaction_dict

        cfg = build_default_redaction_dict()
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(cfg, f)
        f.flush()
        f.close()
        yield f.name
        os.unlink(f.name)

    def test_api_key_in_input_redacted(self, cc_session_id, project_dir, capture_server, redaction_config):
        """API key in tool_input is redacted when redaction.json is present."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_redact1")

        result = run_hook(
            {
                "tool_name": "Bash",
                "session_id": cc_session_id,
                "tool_input": {"api_key": "sk-ant-api03-realsecretkeythatshouldbe"},
                "tool_response": "ok",
            },
            project_dir,
            env={**_capture_env(capture_server), "MYELIN_REDACTION_CONFIG": redaction_config},
        )
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        body = CaptureHandler.captured[0]
        assert body["tool_input"]["api_key"] == "[REDACTED]"

    def test_bearer_in_response_redacted(self, cc_session_id, project_dir, capture_server, redaction_config):
        """Bearer token in tool_response is redacted."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_redact2")

        result = run_hook(
            {
                "tool_name": "Bash",
                "session_id": cc_session_id,
                "tool_input": {"command": "curl"},
                "tool_response": "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.test",
            },
            project_dir,
            env={**_capture_env(capture_server), "MYELIN_REDACTION_CONFIG": redaction_config},
        )
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        body = CaptureHandler.captured[0]
        assert "Bearer" not in body["tool_response"] or "[REDACTED]" in body["tool_response"]

    def test_redact_disabled_via_env(self, cc_session_id, project_dir, capture_server):
        """MYELIN_REDACT=0 disables redaction."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_redact3")

        secret = "sk-ant-api03-realsecretkeythatshouldbe"
        result = run_hook(
            {
                "tool_name": "Bash",
                "session_id": cc_session_id,
                "tool_input": {"key": secret},
                "tool_response": "ok",
            },
            project_dir,
            env={**_capture_env(capture_server), "MYELIN_REDACT": "0"},
        )
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        body = CaptureHandler.captured[0]
        # Secret should pass through unredacted
        assert body["tool_input"]["key"] == secret

    def test_no_config_file_uses_defaults(self, cc_session_id, project_dir, capture_server):
        """Without redaction.json, built-in defaults still redact secrets."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_redact4")

        secret = "sk-ant-api03-realsecretkeythatshouldbe"
        result = run_hook({
            "tool_name": "Bash",
            "session_id": cc_session_id,
            "tool_input": {"key": secret},
            "tool_response": "ok",
        }, project_dir, env=_capture_env(capture_server))
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        body = CaptureHandler.captured[0]
        assert body["tool_input"]["key"] == "[REDACTED]"


class TestDebugMode:
    def test_debug_logging(self, cc_session_id, project_dir):
        result = run_hook(
            {
                "tool_name": "Bash",
                "session_id": cc_session_id,
            },
            project_dir,
            env={"MYELIN_DEBUG": "1"},
        )
        assert result.returncode == 0
        assert "DEBUG" in result.stderr


class TestPathTraversal:
    """Verify session_file_path sanitizes untrusted cc_session_id."""

    def test_traversal_slashes(self, project_dir):
        """../../../etc/passwd should be sanitized to underscores."""
        malicious_id = "../../../etc/passwd"
        sf = session_file(project_dir, malicious_id)
        assert ".." not in sf.name
        assert "/" not in sf.name
        sessions_dir = Path(project_dir) / ".claude" / ".myelin-sessions"
        assert sf.parent == sessions_dir

    def test_traversal_creates_safe_file(self, project_dir):
        """A malicious session ID should still create a file in the sessions dir."""
        malicious_id = "../../evil"
        result = run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": malicious_id,
            "tool_response": {"session_id": "ses_safe"},
        }, project_dir)
        assert result.returncode == 0
        sf = session_file(project_dir, malicious_id)
        sessions_dir = Path(project_dir) / ".claude" / ".myelin-sessions"
        assert sf.parent == sessions_dir
        if sf.exists():
            assert sf.read_text() == "ses_safe"

    def test_normal_id_unchanged(self, project_dir):
        """Normal alphanumeric IDs should pass through unchanged."""
        normal_id = "abc-123_def"
        sf = session_file(project_dir, normal_id)
        assert sf.name == normal_id


class TestMcpJsonLoading:
    """Verify _load_env reads credentials from .mcp.json."""

    def test_loads_from_mcp_json(self):
        """Credentials are derived from .mcp.json when env vars are empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp_config = {
                "mcpServers": {
                    "myelin": {
                        "type": "http",
                        "url": "http://localhost:19876/mcp",
                        "headers": {
                            "Authorization": "Bearer test-key-from-mcp",
                        },
                    }
                }
            }
            mcp_path = os.path.join(tmpdir, ".mcp.json")
            with open(mcp_path, "w") as f:
                json.dump(mcp_config, f)

            result = run_hook(
                {
                    "tool_name": "mcp__myelin__memory_recall",
                    "session_id": "test_mcp_load",
                    "tool_response": {"session_id": "ses_mcp"},
                },
                tmpdir,
                env={
                    "MYELIN_URL": "",
                    "MYELIN_API_KEY": "",
                },
            )
            assert result.returncode == 0
            sf = session_file(tmpdir, "test_mcp_load")
            assert sf.exists()
            assert sf.read_text() == "ses_mcp"

    def test_env_vars_take_precedence(self):
        """Explicit env vars are used even when .mcp.json exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp_config = {
                "mcpServers": {
                    "myelin": {
                        "type": "http",
                        "url": "http://wrong-host/mcp",
                        "headers": {
                            "Authorization": "Bearer wrong-key",
                        },
                    }
                }
            }
            mcp_path = os.path.join(tmpdir, ".mcp.json")
            with open(mcp_path, "w") as f:
                json.dump(mcp_config, f)

            # Explicit env vars should win
            result = run_hook(
                {
                    "tool_name": "mcp__myelin__memory_recall",
                    "session_id": "test_env_precedence",
                    "tool_response": {"session_id": "ses_env"},
                },
                tmpdir,
                # _COMMON_ENV already sets MYELIN_URL/MYELIN_API_KEY
            )
            assert result.returncode == 0

    def test_missing_mcp_json_graceful(self):
        """No .mcp.json and no env vars → missing env error on recall."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_hook(
                {
                    "tool_name": "mcp__myelin__memory_recall",
                    "session_id": "test_no_mcp",
                    "tool_response": {"session_id": "ses_none"},
                },
                tmpdir,
                env={
                    "MYELIN_URL": "",
                    "MYELIN_API_KEY": "",
                },
            )
            assert result.returncode == 2
            assert "MYELIN_URL" in result.stderr


class TestUrlValidation:
    """Verify SSRF protection: reject non-HTTPS URLs for non-localhost."""

    def test_http_remote_url_rejected(self, cc_session_id, project_dir):
        """HTTP URL to a remote host should be rejected (SSRF protection)."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_ssrf")

        result = run_hook(
            {
                "tool_name": "Bash",
                "session_id": cc_session_id,
                "tool_input": {"command": "ls"},
                "tool_response": "file1",
            },
            project_dir,
            env={"MYELIN_URL": "http://evil.com"},
        )
        assert result.returncode == 0
        assert "HTTPS is required" in result.stderr

    def test_http_localhost_url_allowed(self, cc_session_id, project_dir, capture_server):
        """HTTP URL to localhost should still work (dev use)."""
        sf = session_file(project_dir, cc_session_id)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("ses_local")

        result = run_hook(
            {
                "tool_name": "Bash",
                "session_id": cc_session_id,
                "tool_input": {"command": "ls"},
                "tool_response": "file1",
            },
            project_dir,
            env=_capture_env(capture_server),
        )
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1

    def test_mcp_json_http_remote_rejected(self):
        """HTTP remote URL in .mcp.json should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp_config = {
                "mcpServers": {
                    "myelin": {
                        "type": "http",
                        "url": "http://evil.com/mcp",
                        "headers": {
                            "Authorization": "Bearer stolen-key",
                        },
                    }
                }
            }
            mcp_path = os.path.join(tmpdir, ".mcp.json")
            with open(mcp_path, "w") as f:
                json.dump(mcp_config, f)

            result = run_hook(
                {
                    "tool_name": "mcp__myelin__memory_recall",
                    "session_id": "test_mcp_ssrf",
                    "tool_response": {"session_id": "ses_ssrf"},
                },
                tmpdir,
                env={"MYELIN_URL": "", "MYELIN_API_KEY": ""},
            )
            # Should fail because URL was rejected, so env vars remain unset
            assert result.returncode == 2
            assert "MYELIN_URL" in result.stderr


class TestStaleCleanup:
    """Verify stale session files are cleaned up on recall."""

    def test_stale_files_removed(self, cc_session_id, project_dir):
        """Files older than 25 hours are removed on recall."""
        sessions_dir = Path(project_dir) / ".claude" / ".myelin-sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Create a stale file (set mtime to 26 hours ago)
        stale_file = sessions_dir / "old_session"
        stale_file.write_text("ses_old")
        old_time = time.time() - 26 * 3600
        os.utime(stale_file, (old_time, old_time))

        # Create a recent file (should survive)
        recent_file = sessions_dir / "recent_session"
        recent_file.write_text("ses_recent")

        # Trigger recall which runs cleanup
        run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": cc_session_id,
            "tool_response": {"session_id": "ses_new"},
        }, project_dir)

        assert not stale_file.exists(), "stale file should be removed"
        assert recent_file.exists(), "recent file should survive"
        assert session_file(project_dir, cc_session_id).read_text() == "ses_new"
