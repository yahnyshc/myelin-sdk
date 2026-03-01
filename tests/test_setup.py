"""Tests for the myelin-init interactive setup flow."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from myelin_sdk.claude_code.setup import (
    _build_env_content,
    _build_mcp_json,
    _build_settings_json,
    _gitignore_entries_needed,
    init,
)


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    """Run tests in an isolated temporary directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestBuildMcpJson:
    def test_new_file(self, project_dir):
        content, existed = _build_mcp_json("my_key")
        assert not existed
        parsed = json.loads(content)
        assert parsed["mcpServers"]["myelin"]["headers"]["Authorization"] == "Bearer my_key"
        assert parsed["mcpServers"]["myelin"]["type"] == "http"

    def test_existing_file_merges(self, project_dir):
        existing = {"mcpServers": {"other": {"url": "http://other"}}}
        Path(".mcp.json").write_text(json.dumps(existing))

        content, existed = _build_mcp_json("my_key")
        assert existed
        parsed = json.loads(content)
        assert "other" in parsed["mcpServers"]
        assert "myelin" in parsed["mcpServers"]

    def test_invalid_json_overwrites(self, project_dir):
        Path(".mcp.json").write_text("not json")
        content, existed = _build_mcp_json("my_key")
        assert existed
        parsed = json.loads(content)
        assert "myelin" in parsed["mcpServers"]


class TestBuildEnvContent:
    def test_new_env(self, project_dir):
        hooks_dir = Path(".claude/hooks")
        hooks_dir.mkdir(parents=True)

        content, existed = _build_env_content(hooks_dir, "my_key")
        assert not existed
        assert "MYELIN_URL=" in content
        assert "MYELIN_API_KEY=my_key" in content

    def test_existing_env_updates(self, project_dir):
        hooks_dir = Path(".claude/hooks")
        hooks_dir.mkdir(parents=True)
        (hooks_dir / ".env").write_text("MYELIN_URL=old\nOTHER=keep\n")

        content, existed = _build_env_content(hooks_dir, "new_key")
        assert existed
        assert "MYELIN_API_KEY=new_key" in content
        assert "OTHER=keep" in content
        assert "old" not in content


class TestBuildSettingsJson:
    def test_new_settings(self, project_dir):
        command = 'python3 "$CLAUDE_PROJECT_DIR"/.claude/hooks/myelin-capture.py'
        content, existed, already_present = _build_settings_json(command)
        assert not existed
        assert not already_present
        parsed = json.loads(content)
        hooks = parsed["hooks"]["PostToolUse"]
        assert len(hooks) == 1
        assert "myelin-capture" in hooks[0]["hooks"][0]["command"]

    def test_existing_hook_not_duplicated(self, project_dir):
        command = 'python3 "$CLAUDE_PROJECT_DIR"/.claude/hooks/myelin-capture.py'
        settings = {
            "hooks": {
                "PostToolUse": [{
                    "matcher": ".*",
                    "hooks": [{"type": "command", "command": command}],
                }]
            }
        }
        Path(".claude").mkdir()
        Path(".claude/settings.json").write_text(json.dumps(settings))

        content, existed, already_present = _build_settings_json(command)
        assert existed
        assert already_present
        parsed = json.loads(content)
        assert len(parsed["hooks"]["PostToolUse"]) == 1


class TestGitignoreEntries:
    def test_no_gitignore(self, project_dir):
        needed = _gitignore_entries_needed()
        assert ".mcp.json" in needed
        assert ".claude/hooks/.env" in needed

    def test_partial_gitignore(self, project_dir):
        Path(".gitignore").write_text(".mcp.json\n")
        needed = _gitignore_entries_needed()
        assert ".mcp.json" not in needed
        assert ".claude/hooks/.env" in needed

    def test_complete_gitignore(self, project_dir):
        Path(".gitignore").write_text(".mcp.json\n.claude/hooks/.env\n.claude/.myelin-sessions/\n")
        needed = _gitignore_entries_needed()
        assert needed == []


class TestInitFlow:
    def test_all_confirmed(self, project_dir):
        """When user confirms everything, all files are created."""
        # Steps: api_key, .mcp.json, .env, capture hook, redaction config,
        #        settings.json, .gitignore
        inputs = iter(["test_api_key", "y", "y", "y", "y", "y", "y"])
        with patch("builtins.input", side_effect=inputs):
            init()

        assert Path(".mcp.json").exists()
        mcp = json.loads(Path(".mcp.json").read_text())
        assert mcp["mcpServers"]["myelin"]["headers"]["Authorization"] == "Bearer test_api_key"

        assert Path(".claude/hooks/.env").exists()
        env = Path(".claude/hooks/.env").read_text()
        assert "MYELIN_API_KEY=test_api_key" in env

        assert Path(".claude/hooks/myelin-capture.py").exists()
        assert not Path(".claude/hooks/myelin-redact.py").exists()

        assert Path(".claude/hooks/redaction.json").exists()
        redaction = json.loads(Path(".claude/hooks/redaction.json").read_text())
        assert isinstance(redaction["patterns"], list)
        assert isinstance(redaction["sensitive_keys"], list)

        assert Path(".claude/settings.json").exists()
        settings = json.loads(Path(".claude/settings.json").read_text())
        assert "PostToolUse" in settings["hooks"]

        assert Path(".gitignore").exists()
        gitignore = Path(".gitignore").read_text()
        assert ".mcp.json" in gitignore

    def test_all_declined(self, project_dir):
        """When user declines everything, no files are created (except dirs)."""
        inputs = iter(["test_api_key", "n", "n", "n", "n", "n", "n"])
        with patch("builtins.input", side_effect=inputs):
            init()

        assert not Path(".mcp.json").exists()
        assert not Path(".claude/hooks/.env").exists()
        assert not Path(".claude/hooks/myelin-capture.py").exists()
        assert not Path(".claude/hooks/myelin-redact.py").exists()
        # settings.json may not exist since we declined
        if Path(".claude/settings.json").exists():
            settings = json.loads(Path(".claude/settings.json").read_text())
            assert "PostToolUse" not in settings.get("hooks", {})

    def test_selective_confirm(self, project_dir):
        """User can confirm some files and skip others."""
        # .mcp.json=y, .env=n, capture=y, redaction.json=y, settings=y, gitignore=y
        inputs = iter(["test_api_key", "y", "n", "y", "y", "y", "y"])
        with patch("builtins.input", side_effect=inputs):
            init()

        assert Path(".mcp.json").exists()
        assert not Path(".claude/hooks/.env").exists()
        assert Path(".claude/hooks/myelin-capture.py").exists()
