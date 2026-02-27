"""CLI for Myelin setup commands."""

import importlib.resources
import json
import os
import shutil
import stat
import sys
from pathlib import Path

MYELIN_SERVER_URL = "https://myelin.fly.dev"
MYELIN_MCP_URL = f"{MYELIN_SERVER_URL}/mcp"

# -- ANSI styling --------------------------------------------------------------

_NO_COLOR = bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty()


def _sgr(*codes: int) -> str:
    if _NO_COLOR:
        return ""
    return f"\033[{';'.join(str(c) for c in codes)}m"


RESET = _sgr(0)
BOLD = _sgr(1)
DIM = _sgr(2)
GREEN = _sgr(32)
YELLOW = _sgr(33)
CYAN = _sgr(36)
BOLD_CYAN = _sgr(1, 36)
BOLD_GREEN = _sgr(1, 32)
BOLD_YELLOW = _sgr(1, 33)
BOLD_RED = _sgr(1, 31)


def _ok(msg: str) -> None:
    print(f"  {GREEN}✔{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {BOLD_YELLOW}⚠{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {DIM}▸{RESET} {msg}")


def _dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


# -- Helpers -------------------------------------------------------------------


def _install_hook_script(hooks_dir: Path, package_name: str, dest_name: str) -> Path:
    """Copy a hook script from package data and make it executable."""
    source = importlib.resources.files("myelin_sdk.claude_code").joinpath(package_name)
    dest = hooks_dir / dest_name
    with importlib.resources.as_file(source) as src_path:
        shutil.copy2(src_path, dest)
    dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def _ensure_hook_entry(
    hooks_section: dict,
    event: str,
    command: str,
    identifier: str,
    *,
    async_hook: bool = True,
    timeout: int = 10,
) -> bool:
    """Ensure a hook entry exists in settings. Returns True if settings were modified."""
    hook_entry = {
        "matcher": ".*",
        "hooks": [{
            "type": "command",
            "command": command,
            "async": async_hook,
            "timeout": timeout,
        }],
    }

    event_hooks = hooks_section.setdefault(event, [])

    for entry in event_hooks:
        for h in entry.get("hooks", []):
            if identifier in h.get("command", ""):
                return False

    event_hooks.append(hook_entry)
    return True


def _prompt_required(prompt_text: str) -> str:
    """Prompt user for required input, repeat until non-empty."""
    while True:
        value = input(f"  {BOLD_CYAN}?{RESET} {prompt_text}").strip()
        if value:
            return value
        print(f"    {BOLD_RED}This field is required.{RESET}")


def _write_mcp_json(api_key: str) -> None:
    """Write or merge .mcp.json with the Myelin server entry."""
    mcp_path = Path(".mcp.json")
    existed = mcp_path.exists()
    if existed:
        try:
            config = json.loads(mcp_path.read_text())
        except (json.JSONDecodeError, ValueError):
            config = {}
    else:
        config = {}

    servers = config.setdefault("mcpServers", {})
    servers["myelin"] = {
        "type": "http",
        "url": MYELIN_MCP_URL,
        "headers": {
            "Authorization": f"Bearer {api_key}",
        },
    }
    mcp_path.write_text(json.dumps(config, indent=2) + "\n")

    verb = "Updated" if existed else "Created"
    _ok(f"{verb} {BOLD}.mcp.json{RESET}")


def _write_env_file(hooks_dir: Path, api_key: str) -> None:
    """Write or merge .claude/hooks/.env with MYELIN_URL and MYELIN_API_KEY."""
    env_path = hooks_dir / ".env"
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    env_vars = {
        "MYELIN_URL": MYELIN_SERVER_URL,
        "MYELIN_API_KEY": api_key,
    }
    for key, value in env_vars.items():
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}=") or line.startswith(f"export {key}="):
                lines[i] = f"{key}={value}"
                found = True
                break
        if not found:
            lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n")
    _ok(f"Wrote {BOLD}.claude/hooks/.env{RESET} {_dim('(MYELIN_URL, MYELIN_API_KEY)')}")


# -- Main entry point ---------------------------------------------------------


def init():
    """Set up Myelin in the current project interactively."""
    print()
    print(f"  {BOLD_CYAN}╭──────────────────────────────────╮{RESET}")
    print(f"  {BOLD_CYAN}│{RESET}  {BOLD}Myelin{RESET} — interactive setup      {BOLD_CYAN}│{RESET}")
    print(f"  {BOLD_CYAN}╰──────────────────────────────────╯{RESET}")
    print()
    print(f"  This will configure Myelin in {BOLD}{Path.cwd().name}/{RESET}:")
    print()
    print(f"    {DIM}1.{RESET} .mcp.json              {_dim('MCP server connection')}")
    print(f"    {DIM}2.{RESET} .claude/hooks/.env     {_dim('API credentials')}")
    print(f"    {DIM}3.{RESET} .claude/hooks/         {_dim('tool call capture')}")
    print(f"    {DIM}4.{RESET} .claude/settings.json  {_dim('hook registration')}")
    print()

    # 1. Get API key
    api_key = _prompt_required("Myelin API key: ")
    print()

    # 2. Write .mcp.json
    _write_mcp_json(api_key)

    # 3. Install hook script + write .env next to it
    hooks_dir = Path(".claude/hooks")
    hooks_dir.mkdir(parents=True, exist_ok=True)

    _write_env_file(hooks_dir, api_key)

    _install_hook_script(hooks_dir, "capture.py", "myelin-capture.py")
    _ok(f"Installed {BOLD}.claude/hooks/myelin-capture.py{RESET}")

    # Clean up old hooks if upgrading
    for old_name in ("myelin-capture.sh", "myelin-reasoning.py"):
        old_dest = hooks_dir / old_name
        if old_dest.exists():
            old_dest.unlink()
            _info(f"Removed old {old_name}")

    # 4. Update .claude/settings.json
    settings_path = Path(".claude/settings.json")
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
    else:
        settings = {}

    hooks = settings.setdefault("hooks", {})

    if _ensure_hook_entry(
        hooks, "PostToolUse",
        command='python3 "$CLAUDE_PROJECT_DIR"/.claude/hooks/myelin-capture.py',
        identifier="myelin-capture",
        async_hook=True,
        timeout=10,
    ):
        _ok(f"Registered PostToolUse hook in {BOLD}.claude/settings.json{RESET}")

    # Clean up old PreToolUse reasoning hook if present
    pre_hooks = hooks.get("PreToolUse", [])
    new_pre_hooks = [
        entry for entry in pre_hooks
        if not any("myelin-reasoning" in h.get("command", "") for h in entry.get("hooks", []))
    ]
    if len(new_pre_hooks) != len(pre_hooks):
        if new_pre_hooks:
            hooks["PreToolUse"] = new_pre_hooks
        else:
            hooks.pop("PreToolUse", None)
        _info("Removed old PreToolUse reasoning hook")

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    # 5. Success summary
    print()
    print(f"  {BOLD_GREEN}Setup complete!{RESET}")
    print()
    print("  Restart Claude Code to activate Myelin. The MCP server")
    print("  and hooks will load automatically from your project config.")
    print()
