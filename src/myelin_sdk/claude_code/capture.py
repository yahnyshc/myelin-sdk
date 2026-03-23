#!/usr/bin/env python3
"""Myelin PostToolUse / PostToolUseFailure hook — captures tool calls to Myelin.

Reads JSON from stdin provided by Claude Code. Extracts conversation context
(thinking, assistant text, user messages) from the JSONL transcript using offset
tracking and sends it alongside the tool call. For failures, the error message is
always captured (even for investigation tools whose output is normally stripped).
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from myelin_sdk._utils import MAX_CONTEXT_LEN, truncate, validate_base_url
from myelin_sdk.redact import RedactionConfig, redact_dict, redact_string

# -- Hook constants -----------------------------------------------------------

MYELIN_TOOL_PREFIX = "mcp__myelin__"
_ENV_LOADED = False
RECORD = f"{MYELIN_TOOL_PREFIX}record"
FINISH = f"{MYELIN_TOOL_PREFIX}finish"
CAPTURE_TIMEOUT = 5  # seconds
CAPTURE_RETRIES = 2
RETRY_DELAY = 1.0  # seconds

# Investigation tools: capture tool name + input (file paths, patterns) but
# strip the response (large, context-dependent, not useful for workflows).
INVESTIGATION_TOOLS = {"Read", "Glob", "Grep"}


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


def _clear_all_session_files(sessions_dir: str) -> None:
    """Delete all existing session files. Called on start to ensure only one active session."""
    try:
        for name in os.listdir(sessions_dir):
            path = os.path.join(sessions_dir, name)
            try:
                os.remove(path)
            except OSError:
                pass
    except OSError:
        pass


# -- Session file helpers (two-line format: session_id\noffset) ---------------


def _read_session_file(path: str) -> tuple[str, int] | None:
    """Read session file. Returns (session_id, transcript_offset) or None."""
    try:
        with open(path) as f:
            lines = f.read().strip().split("\n")
            sid = lines[0].strip()
            offset = int(lines[1].strip()) if len(lines) > 1 else 0
            return sid, offset
    except (FileNotFoundError, ValueError, IndexError):
        return None


def _write_session_file(path: str, sid: str, offset: int) -> None:
    """Write session file with session_id and transcript offset."""
    try:
        with open(path, "w") as f:
            f.write(f"{sid}\n{offset}")
    except OSError:
        pass


# -- HTTP helper --------------------------------------------------------------


def _post_capture(
    url: str, key: str, payload: dict, retries: int = 0
) -> bool:
    """POST a capture payload to the server. Returns True on success."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/v1/capture",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )

    for attempt in range(retries + 1):
        try:
            urllib.request.urlopen(req, timeout=CAPTURE_TIMEOUT)
            return True
        except (urllib.error.URLError, OSError) as exc:
            if attempt < retries:
                debug(
                    f"capture attempt {attempt + 1} failed: {exc}, retrying"
                )
                time.sleep(RETRY_DELAY)
            else:
                if retries > 0:
                    log(f"capture failed after {attempt + 1} attempts: {exc}")
                else:
                    debug(f"capture failed: {exc}")
    return False


# -- Transcript context extraction --------------------------------------------


def extract_context_from_transcript(
    transcript_path: str,
    offset: int,
) -> tuple[str | None, int]:
    """Extract conversation context from transcript starting at offset.

    Collects assistant thinking, text, and user messages.
    Skips tool_use, tool_result, progress, and file-history-snapshot entries.

    Returns (context_text, new_offset). context_text is None if no content found.
    """
    try:
        with open(transcript_path, "r") as f:
            all_lines = f.readlines()
    except (OSError, IOError):
        return None, offset

    new_offset = len(all_lines)
    if offset >= new_offset:
        return None, new_offset

    session_lines = all_lines[offset:]
    parts: list[str] = []

    for line in session_lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        message = entry.get("message")
        if not message:
            continue

        role = message.get("role")
        if role not in ("assistant", "user"):
            continue

        content = message.get("content", [])

        # User messages: content can be a plain string or array of blocks
        if role == "user":
            if isinstance(content, str):
                text = content.strip()
                if text:
                    parts.append(f"[user] {text}")
                continue
            if isinstance(content, list):
                # Skip tool_result messages
                has_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                )
                if has_tool_result:
                    continue
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            parts.append(f"[user] {text}")
            continue

        # Assistant messages: content is always an array
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "thinking":
                text = block.get("thinking", "").strip()
                if text:
                    parts.append(f"[thinking] {text}")
            elif block_type == "text":
                text = block.get("text", "").strip()
                if text:
                    parts.append(f"[assistant] {text}")
            # Skip tool_use — already captured via PostToolUse

    if not parts:
        return None, new_offset

    return truncate("\n".join(parts), MAX_CONTEXT_LEN), new_offset


# -- Session ID extraction ----------------------------------------------------


def _extract_from_text(text: str):
    """Extract session_id from plain text.

    Handles both formats:
      - 'session_id: ses_abc123\\n...'  (legacy JSON response)
      - 'Session started: ses_abc123\\n...'  (plain-text MCP response)
    """
    for prefix in ("session_id:", "Session started:"):
        if text.startswith(prefix):
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
    """Extract myelin session_id from a start tool response.

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
        # Text might be JSON (MCP tools return JSON strings in content blocks)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

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


# -- Environment loading ------------------------------------------------------


def _load_env() -> None:
    """Derive MYELIN_URL and MYELIN_API_KEY from .mcp.json if not already set."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    if os.environ.get("MYELIN_URL") and os.environ.get("MYELIN_API_KEY"):
        return

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if not project_dir:
        return

    mcp_path = os.path.join(project_dir, ".mcp.json")
    try:
        with open(mcp_path) as f:
            config = json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return

    server = config.get("mcpServers", {}).get("myelin")
    if not isinstance(server, dict):
        return

    # Derive MYELIN_URL from the MCP server URL (strip /mcp suffix)
    url = server.get("url", "")
    if url and not os.environ.get("MYELIN_URL"):
        base_url = url.removesuffix("/mcp")
        try:
            validate_base_url(base_url)
        except ValueError as exc:
            log(f"ignoring URL from .mcp.json: {exc}")
            return
        os.environ["MYELIN_URL"] = base_url

    # Extract API key from Authorization header
    auth = server.get("headers", {}).get("Authorization", "")
    if auth and not os.environ.get("MYELIN_API_KEY"):
        # "Bearer <key>" -> "<key>"
        key = auth.removeprefix("Bearer ").strip()
        if key:
            os.environ["MYELIN_API_KEY"] = key

    if os.environ.get("MYELIN_URL") and os.environ.get("MYELIN_API_KEY"):
        debug(f"loaded credentials from {mcp_path}")


def _load_redaction_config() -> RedactionConfig | None:
    """Load redaction config from env / auto-discovery. Returns None if disabled."""
    try:
        return RedactionConfig.from_env()
    except Exception:
        return None


# -- Main hook entry point ----------------------------------------------------


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

    # Validate URL scheme to prevent sending API keys to unintended servers
    if myelin_url:
        try:
            validate_base_url(myelin_url)
        except ValueError as exc:
            log(str(exc))
            return 0

    # 1. record — persist Myelin session_id
    if tool_name == RECORD:
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
            # Clear all existing session files — only one active session at a time
            _clear_all_session_files(sessions_dir)

            # Store transcript position at record time
            transcript_offset = 0
            transcript_path = data.get("transcript_path") or ""
            if transcript_path:
                try:
                    with open(transcript_path, "r") as tf:
                        transcript_offset = sum(1 for _ in tf)
                except (OSError, IOError):
                    pass

            _write_session_file(session_file, sid, transcript_offset)
            debug(f"session started: {sid}")
        return 0

    # 2. finish — capture trailing context, then clean up
    if tool_name == FINISH:
        session_data = _read_session_file(session_file) if session_file else None

        if session_data and myelin_url and myelin_key:
            myelin_sid, transcript_offset = session_data
            context = None
            transcript_path = data.get("transcript_path") or ""
            if transcript_path:
                context, _ = extract_context_from_transcript(
                    transcript_path, transcript_offset
                )
                if context:
                    debug(f"extracted trailing context ({len(context)} chars)")
                    if (
                        redaction_cfg
                        and redaction_cfg.enabled
                        and redaction_cfg.redact_context
                    ):
                        context = redact_string(context, redaction_cfg)

            if context:
                _post_capture(myelin_url, myelin_key, {
                    "session_id": myelin_sid,
                    "tool_name": "finish",
                    "tool_input": {},
                    "tool_response": "",
                    "context": context,
                    "client_ts": time.time(),
                })
                debug("captured trailing context with finish")

        # Clean up session file
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

    # 4. Skip Claude Code internal tools
    #    Action tools (Edit, Write, Bash, etc.) are fully captured.
    #    Investigation tools (Read, Glob, Grep) are captured input-only.
    #    Everything else is planning, coordination, or navigation noise.
    if tool_name in (
        # Task management & planning
        "TaskCreate", "TaskUpdate", "TaskList", "TaskGet",
        "TodoWrite", "TodoRead", "ToolSearch",
        "EnterPlanMode", "ExitPlanMode", "AskUserQuestion",
        # Read-only / navigation (NOT Read, Glob, Grep — those are
        # captured input-only via INVESTIGATION_TOOLS)
        "LS", "NotebookRead", "LSP",
        # Agent coordination
        "Agent", "TeamCreate", "TeamDelete", "SendMessage",
        "EnterWorktree", "Skill",
        # Process management
        "TaskOutput", "TaskStop",
    ):
        return 0

    # 5. No active session — nothing to do
    if not session_file:
        return 0
    session_data = _read_session_file(session_file)
    if not session_data:
        return 0
    myelin_sid, transcript_offset = session_data

    # 6. Capture the tool call
    if not myelin_url or not myelin_key:
        debug("skipping capture: MYELIN_URL or MYELIN_API_KEY not set")
        return 0

    tool_input = data.get("tool_input", {})

    # Detect failures: PostToolUseFailure sends "error" without "tool_response"
    is_error = "error" in data and "tool_response" not in data

    # For errors, always capture the error message (even for investigation tools
    # whose output is normally stripped — error messages are valuable signal).
    if is_error:
        tool_response = str(data.get("error", ""))
    elif tool_name in INVESTIGATION_TOOLS:
        # Investigation tools: keep input (file paths, patterns) but strip output
        tool_response = ""
    else:
        tool_response = str(data.get("tool_response", ""))

    # Redact before truncate
    if redaction_cfg and redaction_cfg.enabled:
        if redaction_cfg.redact_tool_input:
            tool_input = redact_dict(tool_input, redaction_cfg)
        if redaction_cfg.redact_tool_response:
            tool_response = redact_string(tool_response, redaction_cfg)

    tool_response = truncate(tool_response)

    context = None
    new_offset = transcript_offset
    transcript_path = data.get("transcript_path") or ""
    if transcript_path:
        context, new_offset = extract_context_from_transcript(
            transcript_path, transcript_offset
        )
        if context:
            debug(f"extracted context ({len(context)} chars)")

    if (
        context
        and redaction_cfg
        and redaction_cfg.enabled
        and redaction_cfg.redact_context
    ):
        context = redact_string(context, redaction_cfg)

    capture_payload: dict = {
        "session_id": myelin_sid,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_response": tool_response,
        "client_ts": time.time(),
    }
    if context:
        capture_payload["context"] = context
    if is_error:
        capture_payload["is_error"] = True

    if _post_capture(myelin_url, myelin_key, capture_payload, CAPTURE_RETRIES):
        debug(f"captured {tool_name}")
        # Update offset only if it changed
        if new_offset != transcript_offset and session_file:
            _write_session_file(session_file, myelin_sid, new_offset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
