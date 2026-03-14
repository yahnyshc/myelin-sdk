"""Tests for myelin-sync CLI and client.sync_workflows()."""

import http.server
import json
import os
import subprocess
import sys
import threading

import httpx
import pytest

from myelin_sdk.client import MyelinClient
from myelin_sdk.sync import (
    _extract_description,
    _relative_path,
    collect_files,
)


# -- Unit tests: description extraction --------------------------------------


class TestExtractDescription:
    def test_heading(self):
        assert _extract_description("# Deploy to Production\nsteps...", "f.md") == "Deploy to Production"

    def test_heading_with_leading_whitespace(self):
        content = "\n\n# My Procedure\nstuff"
        assert _extract_description(content, "f.md") == "My Procedure"

    def test_fallback_to_filename(self):
        assert _extract_description("No heading here", "deploy-hotfix.md") == "Deploy Hotfix"

    def test_fallback_underscore(self):
        assert _extract_description("plain text", "run_tests.md") == "Run Tests"

    def test_ignores_h2(self):
        # Only # (h1) is used, not ##
        assert _extract_description("## Sub heading\ntext", "backup.md") == "Backup"


# -- Unit tests: collect_files -----------------------------------------------


class TestCollectFiles:
    def test_explicit_files(self, tmp_path):
        p = tmp_path / "proc.md"
        p.write_text("# My Proc\nDo stuff")
        files = collect_files([str(p)], None)
        assert len(files) == 1
        assert files[0]["description"] == "My Proc"
        assert files[0]["content"] == "# My Proc\nDo stuff"

    def test_directory(self, tmp_path):
        d = tmp_path / "procs"
        d.mkdir()
        (d / "a.md").write_text("# Alpha\ncontent")
        (d / "b.md").write_text("# Beta\ncontent")
        (d / "skip.txt").write_text("not markdown")
        files = collect_files(None, str(d))
        assert len(files) == 2
        names = {f["description"] for f in files}
        assert names == {"Alpha", "Beta"}

    def test_empty_file_skipped(self, tmp_path):
        p = tmp_path / "empty.md"
        p.write_text("   \n  ")
        files = collect_files([str(p)], None)
        assert len(files) == 0

    def test_missing_directory(self, tmp_path):
        files = collect_files(None, str(tmp_path / "nonexistent"))
        assert len(files) == 0

    def test_relative_path(self, tmp_path):
        root = str(tmp_path)
        p = tmp_path / "sub" / "proc.md"
        p.parent.mkdir()
        p.write_text("# Proc\ncontent")
        rel = _relative_path(str(p), root)
        assert rel == os.path.join("sub", "proc.md")


# -- Client integration: sync_workflows --------------------------------------


class TestClientSyncWorkflows:
    async def test_sync_workflows(self):
        requests_sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests_sent.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "details": [
                        {"path": "procs/deploy.md", "status": "created", "workflow_id": "wf_1"},
                        {"path": "procs/rollback.md", "status": "unchanged", "workflow_id": "wf_2"},
                    ],
                    "created": 1,
                    "updated": 0,
                    "unchanged": 1,
                },
            )

        client = MyelinClient(api_key="test-key", base_url="https://test.myelin.dev")
        client._http = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://test.myelin.dev",
            headers={"Authorization": "Bearer test-key"},
        )

        result = await client.sync_workflows([
            {"path": "procs/deploy.md", "content": "# Deploy\nsteps", "description": "Deploy"},
            {"path": "procs/rollback.md", "content": "# Rollback\nsteps", "description": "Rollback"},
        ])

        assert result.created == 1
        assert result.unchanged == 1
        assert result.updated == 0
        assert len(result.details) == 2
        assert result.details[0].status == "created"
        assert result.details[0].workflow_id == "wf_1"

        body = requests_sent[0]
        assert len(body["files"]) == 2
        assert body["files"][0]["path"] == "procs/deploy.md"

        await client.close()

    async def test_sync_workflows_error(self):
        client = MyelinClient(api_key="test-key", base_url="https://test.myelin.dev")
        client._http = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(401, json={"error": "Unauthorized"})
            ),
            base_url="https://test.myelin.dev",
            headers={"Authorization": "Bearer test-key"},
        )

        from myelin_sdk.errors import MyelinAPIError
        with pytest.raises(MyelinAPIError):
            await client.sync_workflows([{"path": "a.md", "content": "x", "description": "x"}])

        await client.close()


# -- Subprocess test: CLI end-to-end -----------------------------------------


class TestSyncCLI:
    """Run sync.py as a subprocess (like test_capture.py tests the hook)."""

    def _run_sync(self, args, env_extra=None, cwd=None):
        env = os.environ.copy()
        # Clear any existing creds to start clean
        env.pop("MYELIN_URL", None)
        env.pop("MYELIN_API_KEY", None)
        env.pop("CLAUDE_PROJECT_DIR", None)
        if env_extra:
            env.update(env_extra)
        result = subprocess.run(
            [sys.executable, "-m", "myelin_sdk.sync"] + args,
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
            timeout=30,
        )
        return result

    def test_no_files_exits_1(self, tmp_path):
        result = self._run_sync(
            ["--dir", str(tmp_path / "empty")],
            env_extra={"MYELIN_URL": "https://x.dev", "MYELIN_API_KEY": "k"},
        )
        assert result.returncode == 1
        assert "No procedures" in result.stdout or "not found" in result.stderr

    def test_sync_with_server(self, tmp_path):
        """Spin up a local HTTP server and sync files to it."""
        # Create procedure files
        proc_dir = tmp_path / "procs"
        proc_dir.mkdir()
        (proc_dir / "deploy.md").write_text("# Deploy App\n\n1. Build\n2. Push\n3. Verify")
        (proc_dir / "rollback.md").write_text("# Rollback\n\nRevert last deploy")

        # Set up a simple HTTP server that returns a sync response
        server_response = {
            "details": [
                {"path": "procs/deploy.md", "status": "created", "workflow_id": "wf_1"},
                {"path": "procs/rollback.md", "status": "updated", "workflow_id": "wf_2"},
            ],
            "created": 1,
            "updated": 1,
            "unchanged": 0,
        }

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                # Verify request structure
                assert self.path == "/v1/workflows/sync"
                assert "files" in body
                assert self.headers["Authorization"] == "Bearer test-key"

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(server_response).encode())

            def log_message(self, *args):
                pass  # suppress logs

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        try:
            result = self._run_sync(
                ["--dir", str(proc_dir)],
                env_extra={
                    "MYELIN_URL": f"http://127.0.0.1:{port}",
                    "MYELIN_API_KEY": "test-key",
                },
            )
            assert result.returncode == 0
            assert "Synced 2 procedures" in result.stdout
            assert "1 new" in result.stdout
            assert "1 updated" in result.stdout
        finally:
            server.server_close()
            thread.join(timeout=5)

    def test_explicit_files(self, tmp_path):
        """Sync specific files by path."""
        p = tmp_path / "hotfix.md"
        p.write_text("# Hotfix Procedure\n\nApply patch")

        server_response = {
            "details": [{"path": "hotfix.md", "status": "created", "workflow_id": "wf_1"}],
            "created": 1,
            "updated": 0,
            "unchanged": 0,
        }

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                assert len(body["files"]) == 1
                assert body["files"][0]["description"] == "Hotfix Procedure"

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(server_response).encode())

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        try:
            result = self._run_sync(
                [str(p)],
                env_extra={
                    "MYELIN_URL": f"http://127.0.0.1:{port}",
                    "MYELIN_API_KEY": "test-key",
                },
            )
            assert result.returncode == 0
            assert "Synced 1 procedures" in result.stdout
            assert "1 new" in result.stdout
        finally:
            server.server_close()
            thread.join(timeout=5)

    def test_credentials_from_mcp_json(self, tmp_path):
        """Credentials loaded from .mcp.json when env vars not set."""
        proc_dir = tmp_path / "procs"
        proc_dir.mkdir()
        (proc_dir / "test.md").write_text("# Test\ncontent")

        server_response = {
            "details": [{"path": "procs/test.md", "status": "unchanged"}],
            "created": 0,
            "updated": 0,
            "unchanged": 1,
        }

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(server_response).encode())

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]

        # Write .mcp.json
        mcp_config = {
            "mcpServers": {
                "myelin": {
                    "url": f"http://127.0.0.1:{port}/mcp",
                    "headers": {"Authorization": "Bearer from-mcp-json"},
                }
            }
        }
        (tmp_path / ".mcp.json").write_text(json.dumps(mcp_config))

        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        try:
            result = self._run_sync(
                ["--dir", str(proc_dir)],
                env_extra={"CLAUDE_PROJECT_DIR": str(tmp_path)},
            )
            assert result.returncode == 0
            assert "Synced 1 procedures" in result.stdout
        finally:
            server.server_close()
            thread.join(timeout=5)

    def test_no_credentials_exits_1(self, tmp_path):
        """Exits with error when no credentials available."""
        p = tmp_path / "proc.md"
        p.write_text("# Proc\ncontent")

        result = self._run_sync([str(p)])
        assert result.returncode == 1
        assert "MYELIN_URL" in result.stderr
