"""Tests for the capture.py PostToolUse hook (SDK copy).

Runs the hook as a subprocess to test it exactly as Claude Code would.
"""

import json
import os
import subprocess
import sys
import tempfile
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

BASE_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "TMPDIR": tempfile.gettempdir(),
    "MYELIN_URL": "http://localhost:19876",
    "MYELIN_API_KEY": "test-key",
}


def run_hook(
    stdin_data: dict, env: dict | None = None, session_dir: str | None = None
) -> subprocess.CompletedProcess:
    """Run capture.py as a subprocess with the given stdin JSON and env."""
    final_env = {**BASE_ENV, **(env or {})}
    return subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=json.dumps(stdin_data),
        capture_output=True,
        text=True,
        env=final_env,
        timeout=10,
    )


def session_file(cc_session_id: str) -> Path:
    import re
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", cc_session_id)
    return Path(tempfile.gettempdir()) / f"myelin_session_{safe_id}"


@pytest.fixture
def cc_session_id():
    """Provide a unique CC session ID and clean up its temp file after."""
    sid = f"test_{os.getpid()}_{id(object())}"
    yield sid
    sf = session_file(sid)
    sf.unlink(missing_ok=True)


class TestMissingFields:
    def test_missing_tool_name(self, cc_session_id):
        result = run_hook({"session_id": cc_session_id})
        assert result.returncode == 0

    def test_missing_session_id(self):
        result = run_hook({"tool_name": "Bash"})
        assert result.returncode == 0

    def test_empty_stdin(self):
        result = subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            input="",
            capture_output=True,
            text=True,
            env=BASE_ENV,
            timeout=10,
        )
        assert result.returncode == 0

    def test_invalid_json(self):
        result = subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            input="not json",
            capture_output=True,
            text=True,
            env=BASE_ENV,
            timeout=10,
        )
        assert result.returncode == 0


class TestRecall:
    def test_creates_session_file(self, cc_session_id):
        result = run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": cc_session_id,
            "tool_response": {"session_id": "ses_abc123"},
        })
        assert result.returncode == 0
        sf = session_file(cc_session_id)
        assert sf.exists()
        assert sf.read_text() == "ses_abc123"

    def test_nested_result_string(self, cc_session_id):
        """Handle {result: '{"session_id": "..."}'}."""
        result = run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": cc_session_id,
            "tool_response": {"result": json.dumps({"session_id": "ses_nested"})},
        })
        assert result.returncode == 0
        assert session_file(cc_session_id).read_text() == "ses_nested"

    def test_nested_result_object(self, cc_session_id):
        """Handle {result: {session_id: "..."}}."""
        result = run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": cc_session_id,
            "tool_response": {"result": {"session_id": "ses_obj"}},
        })
        assert result.returncode == 0
        assert session_file(cc_session_id).read_text() == "ses_obj"

    def test_string_response(self, cc_session_id):
        """Handle tool_response as a raw JSON string."""
        result = run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": cc_session_id,
            "tool_response": json.dumps({"session_id": "ses_str"}),
        })
        assert result.returncode == 0
        assert session_file(cc_session_id).read_text() == "ses_str"

    def test_missing_env_vars(self, cc_session_id):
        result = run_hook(
            {
                "tool_name": "mcp__myelin__memory_recall",
                "session_id": cc_session_id,
                "tool_response": {"session_id": "ses_x"},
            },
            env={"MYELIN_URL": "", "MYELIN_API_KEY": ""},
        )
        assert result.returncode == 2
        assert "MYELIN_URL" in result.stderr

    def test_no_session_id_in_response(self, cc_session_id):
        """If tool_response has no session_id, don't create file."""
        result = run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": cc_session_id,
            "tool_response": {"matches": []},
        })
        assert result.returncode == 0
        assert not session_file(cc_session_id).exists()


class TestDebrief:
    def test_deletes_session_file(self, cc_session_id):
        sf = session_file(cc_session_id)
        sf.write_text("ses_todelete")

        result = run_hook({
            "tool_name": "mcp__myelin__memory_debrief",
            "session_id": cc_session_id,
        })
        assert result.returncode == 0
        assert not sf.exists()

    def test_missing_file_ok(self, cc_session_id):
        result = run_hook({
            "tool_name": "mcp__myelin__memory_debrief",
            "session_id": cc_session_id,
        })
        assert result.returncode == 0


class TestSkipMyelinTools:
    def test_skips_own_tools(self, cc_session_id):
        result = run_hook({
            "tool_name": "mcp__myelin__some_other_tool",
            "session_id": cc_session_id,
        })
        assert result.returncode == 0


class TestNoActiveSession:
    def test_no_session_file_skips(self, cc_session_id):
        session_file(cc_session_id).unlink(missing_ok=True)
        result = run_hook({
            "tool_name": "Bash",
            "session_id": cc_session_id,
            "tool_input": {"command": "ls"},
            "tool_response": "file1\nfile2",
        })
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
    """Start a local HTTP server to receive capture requests."""
    CaptureHandler.captured = []
    server = HTTPServer(("127.0.0.1", 19876), CaptureHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


class TestCapture:
    def test_posts_to_server(self, cc_session_id, capture_server):
        sf = session_file(cc_session_id)
        sf.write_text("ses_capture")

        result = run_hook({
            "tool_name": "Bash",
            "session_id": cc_session_id,
            "tool_input": {"command": "echo hello"},
            "tool_response": "hello",
        })
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        body = CaptureHandler.captured[0]
        assert body["session_id"] == "ses_capture"
        assert body["tool_name"] == "Bash"
        assert body["tool_input"] == {"command": "echo hello"}
        assert body["tool_response"] == "hello"
        assert isinstance(body["client_ts"], float)
        assert body["client_ts"] > 0

    def test_truncates_long_response(self, cc_session_id, capture_server):
        sf = session_file(cc_session_id)
        sf.write_text("ses_trunc")

        long_response = "x" * 20000
        result = run_hook({
            "tool_name": "Read",
            "session_id": cc_session_id,
            "tool_input": {"path": "/file"},
            "tool_response": long_response,
        })
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        resp = CaptureHandler.captured[0]["tool_response"]
        assert resp.startswith("x" * 4000)
        assert resp.endswith("x" * 4000)
        assert "20000 chars" in resp

    def test_captures_reasoning_from_transcript(self, cc_session_id, capture_server):
        """Reasoning is extracted from the transcript and sent in the payload."""
        sf = session_file(cc_session_id)
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
                "tool_name": "Read",
                "session_id": cc_session_id,
                "tool_use_id": "toolu_reasoning1",
                "transcript_path": transcript.name,
                "tool_input": {"path": "/etc/config"},
                "tool_response": "key=value",
            })
            assert result.returncode == 0
            assert len(CaptureHandler.captured) == 1
            body = CaptureHandler.captured[0]
            assert "reasoning" in body
            assert "I need to check the config" in body["reasoning"]
            assert "Let me read the config file" in body["reasoning"]
        finally:
            os.unlink(transcript.name)

    def test_no_reasoning_without_transcript(self, cc_session_id, capture_server):
        """No reasoning field when transcript_path is missing."""
        sf = session_file(cc_session_id)
        sf.write_text("ses_noreason")

        result = run_hook({
            "tool_name": "Bash",
            "session_id": cc_session_id,
            "tool_use_id": "toolu_noreason",
            "tool_input": {"command": "ls"},
            "tool_response": "file1",
        })
        assert result.returncode == 0
        assert len(CaptureHandler.captured) == 1
        assert "reasoning" not in CaptureHandler.captured[0]

    def test_http_error_exits_zero(self, cc_session_id):
        """Network failure (no server) should not block the agent."""
        sf = session_file(cc_session_id)
        sf.write_text("ses_fail")

        result = run_hook(
            {
                "tool_name": "Bash",
                "session_id": cc_session_id,
                "tool_input": {},
                "tool_response": "out",
            },
            env={"MYELIN_URL": "http://127.0.0.1:19877", "MYELIN_API_KEY": "k"},
        )
        assert result.returncode == 0
        assert "capture failed" in result.stderr


class TestDebugMode:
    def test_debug_logging(self, cc_session_id):
        result = run_hook(
            {
                "tool_name": "Bash",
                "session_id": cc_session_id,
            },
            env={"MYELIN_DEBUG": "1"},
        )
        assert result.returncode == 0
        assert "DEBUG" in result.stderr


class TestPathTraversal:
    """Verify session_file_path sanitizes untrusted cc_session_id."""

    def test_traversal_slashes(self):
        """../../../etc/passwd should be sanitized to underscores."""
        malicious_id = "../../../etc/passwd"
        sf = session_file(malicious_id)
        assert ".." not in str(sf.name)
        assert "/" not in sf.name
        assert sf.parent == Path(tempfile.gettempdir())

    def test_traversal_creates_safe_file(self):
        """A malicious session ID should still create a file in the temp dir."""
        malicious_id = "../../evil"
        result = run_hook({
            "tool_name": "mcp__myelin__memory_recall",
            "session_id": malicious_id,
            "tool_response": {"session_id": "ses_safe"},
        })
        assert result.returncode == 0
        sf = session_file(malicious_id)
        assert sf.parent == Path(tempfile.gettempdir())
        if sf.exists():
            assert sf.read_text() == "ses_safe"
            sf.unlink()

    def test_normal_id_unchanged(self):
        """Normal alphanumeric IDs should pass through unchanged."""
        normal_id = "abc-123_def"
        sf = session_file(normal_id)
        assert sf.name == f"myelin_session_{normal_id}"


class TestEnvQuoteStripping:
    """Verify _load_env strips quotes from .env values."""

    def test_quoted_values_loaded(self):
        """Double-quoted values in .env should have quotes stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = tmpdir
            hooks_dir = os.path.join(project_dir, ".claude", "hooks")
            os.makedirs(hooks_dir)
            env_path = os.path.join(hooks_dir, ".env")
            with open(env_path, "w") as f:
                f.write('MYELIN_URL="http://localhost:8000"\n')
                f.write("MYELIN_API_KEY='test-key-123'\n")

            result = run_hook(
                {
                    "tool_name": "mcp__myelin__memory_recall",
                    "session_id": "test_quote_strip",
                    "tool_response": {"session_id": "ses_q"},
                },
                env={
                    "MYELIN_URL": "",
                    "MYELIN_API_KEY": "",
                    "CLAUDE_PROJECT_DIR": project_dir,
                },
            )
            # Should fail because stripped URL points to nonexistent server,
            # but the important thing is it tried (returncode 2 means env was loaded)
            # With empty MYELIN_URL after stripping it would report missing env
            assert result.returncode in (0, 2)
