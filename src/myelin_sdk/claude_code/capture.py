#!/usr/bin/env python3
"""Myelin PostToolUse hook — captures tool calls and reasoning to the Myelin server.

Reads JSON from stdin provided by Claude Code. Extracts agent reasoning from the
JSONL transcript and sends it alongside the tool call.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from myelin_sdk._utils import truncate
from myelin_sdk.redact import RedactionConfig, redact_dict, redact_string

# -- Hook constants -----------------------------------------------------------

MYELIN_TOOL_PREFIX = "mcp__myelin__"
_ENV_LOADED = False
RECALL = f"{MYELIN_TOOL_PREFIX}memory_recall"
FINISH = f"{MYELIN_TOOL_PREFIX}memory_finish"
CAPTURE_TIMEOUT = 5  # seconds
CAPTURE_RETRIES = 2
RETRY_DELAY = 1.0  # seconds


def log(msg: str) -> None:
    print(f"[myelin] {msg}", file=sys.stderr)


def debug(msg: str) -> None:
    if os.environ.get("MYELIN_DEBUG") == "1":
        log(f"DEBUG: {msg}")


_SAFE_ID = re.compile(r"[^a-zA-Z0-9_\-]")


_SESSIONS_DIR_NAME = ".myelin-sessions"


def session_file_path(cc_session_id: str) -> str | None:
    """Return path to session tracking file, or None if CLAUDE_PROJECT_DIR is unset."""
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if not project_dir:
        return None
    safe_id = _SAFE_ID.sub("_", cc_session_id)
    return os.path.join(project_dir, ".claude", _SESSIONS_DIR_NAME, safe_id)


def _cleanup_stale_session_files(sessions_dir: str, max_age_hours: int = 25) -> None:
    """Remove session files older than max_age_hours. Best-effort."""
    cutoff = time.time() - max_age_hours * 3600
    try:
        for name in os.listdir(sessions_dir):
            path = os.path.join(sessions_dir, name)
            try:
                if os.stat(path).st_mtime < cutoff:
                    os.remove(path)
            except OSError:
                pass
    except OSError:
        pass


def extract_reasoning_from_transcript(transcript_path: str, tool_use_id: str) -> str | None:
    """Parse JSONL transcript to find reasoning for a tool_use_id.

    Looks for the assistant message containing the tool_use block with the given id,
    then collects all thinking and text blocks from that same message.
    """
    lines: list[str] = []
    try:
        with open(transcript_path, "r") as f:
            all_lines = f.readlines()
            lines = all_lines[-100:]
    except (OSError, IOError):
        return None

    # Find the message containing our tool_use_id
    target_message_id = None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        message = entry.get("message")
        if not message or message.get("role") != "assistant":
            continue

        content = message.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if isinstance(block, dict) and block.get("id") == tool_use_id:
                target_message_id = message.get("id")
                break

        if target_message_id:
            break

    if not target_message_id:
        debug(f"no transcript entry found for tool_use_id={tool_use_id}")
        return None

    # Collect thinking and text blocks from ALL entries with that message id.
    # Claude Code splits one assistant message across multiple JSONL lines.
    thinking_parts: list[str] = []
    text_parts: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        message = entry.get("message")
        if not message or message.get("id") != target_message_id:
            continue

        content = message.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "thinking":
                thinking_text = block.get("thinking", "").strip()
                if thinking_text:
                    thinking_parts.append(thinking_text)
            elif block_type == "text":
                text_text = block.get("text", "").strip()
                if text_text:
                    text_parts.append(text_text)

    if not thinking_parts and not text_parts:
        debug(f"no reasoning content for message_id={target_message_id}")
        return None

    parts = []
    if thinking_parts:
        parts.extend(thinking_parts)
    if text_parts:
        if parts:
            parts.append("")  # blank line separator
        parts.extend(text_parts)

    return "\n".join(parts)


def _extract_from_text(text: str):
    """Extract session_id from plain text like 'session_id: ses_abc123\\n...'"""
    if text.startswith("session_id:"):
        sid = text.split("\n", 1)[0].split(":", 1)[1].strip()
        if sid:
            return sid
    return None


def _extract_text_from_content_blocks(data):
    """Extract concatenated text from MCP content block list.

    Claude Code sends MCP tool responses as [{type: 'text', text: '...'}].
    """
    if not isinstance(data, list):
        return None
    parts = []
    for block in data:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts) if parts else None


def extract_session_id(tool_response):
    """Extract myelin session_id from a recall tool response.

    The response can arrive in several shapes:
      - Plain text starting with "session_id: <id>"
      - MCP content blocks: [{"type": "text", "text": "session_id: ..."}]
      - A JSON string encoding either of the above
      - A dict with "session_id" or "result" wrapping any of the above
    """
    data = tool_response

    # String — try plain text, then JSON-parse
    if isinstance(data, str):
        sid = _extract_from_text(data)
        if sid:
            return sid
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None

    # Content blocks list: [{"type": "text", "text": "..."}]
    text = _extract_text_from_content_blocks(data)
    if text:
        sid = _extract_from_text(text)
        if sid:
            return sid

    if not isinstance(data, dict):
        return None

    # Unwrap "result" wrapper
    if "result" in data:
        data = data["result"]
        if isinstance(data, str):
            sid = _extract_from_text(data)
            if sid:
                return sid
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return None
        text = _extract_text_from_content_blocks(data)
        if text:
            sid = _extract_from_text(text)
            if sid:
                return sid
        if not isinstance(data, dict):
            return None

    return data.get("session_id")


def _load_env() -> None:
    """Load .claude/hooks/.env if it exists and env vars aren't already set."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    if os.environ.get("MYELIN_URL") and os.environ.get("MYELIN_API_KEY"):
        return

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if not project_dir:
        return

    env_path = os.path.join(project_dir, ".claude", "hooks", ".env")
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                if key and not os.environ.get(key):
                    os.environ[key] = value
        debug(f"loaded env from {env_path}")
    except FileNotFoundError:
        pass


def _load_redaction_config() -> RedactionConfig | None:
    """Load redaction config from env / auto-discovery. Returns None if disabled."""
    try:
        return RedactionConfig.from_env()
    except Exception:
        return None


def main() -> int:
    try:
        data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, TypeError):
        return 0

    tool_name = data.get("tool_name") or ""
    cc_session_id = data.get("session_id") or ""

    if not tool_name or not cc_session_id:
        return 0

    _load_env()
    redaction_cfg = _load_redaction_config()
    debug(f"tool={tool_name}")
    session_file = session_file_path(cc_session_id)

    myelin_url = os.environ.get("MYELIN_URL", "")
    myelin_key = os.environ.get("MYELIN_API_KEY", "")

    # 1. recall — persist Myelin session_id
    if tool_name == RECALL:
        if not myelin_url or not myelin_key:
            log(
                "MYELIN_URL and MYELIN_API_KEY must be set. "
                "Run: export MYELIN_URL=... MYELIN_API_KEY=..."
            )
            return 2

        sid = extract_session_id(data.get("tool_response"))
        if sid and session_file:
            sessions_dir = os.path.dirname(session_file)
            os.makedirs(sessions_dir, exist_ok=True)
            _cleanup_stale_session_files(sessions_dir)
            with open(session_file, "w") as f:
                f.write(sid)
            debug(f"session started: {sid}")
        return 0

    # 2. finish — clean up
    if tool_name == FINISH:
        if session_file:
            try:
                os.remove(session_file)
            except FileNotFoundError:
                pass
        debug("session file cleaned up")
        return 0

    # 3. Skip Myelin's own tools
    if tool_name.startswith(MYELIN_TOOL_PREFIX):
        return 0

    # 4. No active session — nothing to do
    if not session_file:
        return 0
    try:
        with open(session_file) as f:
            myelin_sid = f.read().strip()
    except FileNotFoundError:
        return 0

    # 5. Capture the tool call
    if not myelin_url or not myelin_key:
        debug("skipping capture: MYELIN_URL or MYELIN_API_KEY not set")
        return 0

    tool_input = data.get("tool_input", {})
    tool_response = str(data.get("tool_response", ""))

    # Redact before truncate
    if redaction_cfg and redaction_cfg.enabled:
        if redaction_cfg.redact_tool_input:
            tool_input = redact_dict(tool_input, redaction_cfg)
        if redaction_cfg.redact_tool_response:
            tool_response = redact_string(tool_response, redaction_cfg)

    tool_response = truncate(tool_response)

    # Extract reasoning directly from the transcript
    reasoning = None
    tool_use_id = data.get("tool_use_id") or ""
    transcript_path = data.get("transcript_path") or ""
    if tool_use_id and transcript_path:
        reasoning = extract_reasoning_from_transcript(transcript_path, tool_use_id)
        if reasoning:
            debug(f"extracted reasoning ({len(reasoning)} chars) for {tool_use_id}")

    if reasoning and redaction_cfg and redaction_cfg.enabled and redaction_cfg.redact_reasoning:
        reasoning = redact_string(reasoning, redaction_cfg)

    capture_payload: dict = {
        "session_id": myelin_sid,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_response": tool_response,
        "client_ts": time.time(),
    }
    if reasoning:
        capture_payload["reasoning"] = reasoning

    payload = json.dumps(capture_payload).encode()

    req = urllib.request.Request(
        f"{myelin_url}/v1/capture",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {myelin_key}",
        },
        method="POST",
    )

    for attempt in range(CAPTURE_RETRIES + 1):
        try:
            urllib.request.urlopen(req, timeout=CAPTURE_TIMEOUT)
            debug(f"captured {tool_name}")
            break
        except (urllib.error.URLError, OSError) as exc:
            if attempt < CAPTURE_RETRIES:
                debug(f"capture attempt {attempt + 1} failed for {tool_name}: {exc}, retrying")
                time.sleep(RETRY_DELAY)
            else:
                log(f"capture failed for {tool_name} after {attempt + 1} attempts: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
