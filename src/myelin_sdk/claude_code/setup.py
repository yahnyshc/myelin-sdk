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


def _skip(msg: str) -> None:
    print(f"  {DIM}─{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {BOLD_YELLOW}⚠{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {DIM}▸{RESET} {msg}")


def _dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


# -- Helpers -------------------------------------------------------------------


def _confirm(prompt: str) -> bool:
    """Ask a y/N confirmation question. Returns True for 'y'."""
    try:
        answer = input(f"  {BOLD_CYAN}?{RESET} {prompt} {DIM}[y/N]{RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer == "y"


def _prompt_required(prompt_text: str) -> str:
    """Prompt user for required input, repeat until non-empty."""
    while True:
        value = input(f"  {BOLD_CYAN}?{RESET} {prompt_text}").strip()
        if value:
            return value
        print(f"    {BOLD_RED}This field is required.{RESET}")


def _show_header(path: str) -> None:
    """Print a file preview header."""
    width = max(40, len(path) + 4)
    print(f"\n  {DIM}{'─' * 2} {RESET}{BOLD}{path}{RESET} {DIM}{'─' * (width - len(path) - 4)}{RESET}")


def _show_content(text: str, *, max_lines: int = 20) -> None:
    """Print indented content, truncating if too long."""
    lines = text.splitlines()
    for line in lines[:max_lines]:
        print(f"  {DIM}│{RESET} {line}")
    if len(lines) > max_lines:
        print(f"  {DIM}│ ... ({len(lines)} lines total){RESET}")


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


# -- File builders -------------------------------------------------------------


def _build_mcp_json(api_key: str) -> tuple[str, bool]:
    """Build .mcp.json content. Returns (content, existed)."""
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
    return json.dumps(config, indent=2) + "\n", existed


def _build_env_content(hooks_dir: Path, api_key: str) -> tuple[str, bool]:
    """Build .env content. Returns (content, existed)."""
    env_path = hooks_dir / ".env"
    existed = env_path.exists()
    lines: list[str] = []
    if existed:
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

    return "\n".join(lines) + "\n", existed


def _build_settings_json(command: str) -> tuple[str, bool, bool]:
    """Build .claude/settings.json content.

    Returns (content, existed, hook_already_present).
    """
    settings_path = Path(".claude/settings.json")
    existed = settings_path.exists()
    if existed:
        settings = json.loads(settings_path.read_text())
    else:
        settings = {}

    hooks = settings.setdefault("hooks", {})

    modified = _ensure_hook_entry(
        hooks, "PostToolUse",
        command=command,
        identifier="myelin-capture",
        async_hook=True,
        timeout=10,
    )

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

    return json.dumps(settings, indent=2) + "\n", existed, not modified


def _gitignore_entries_needed() -> list[str]:
    """Return .gitignore entries that should be added."""
    needed = []
    gitignore_path = Path(".gitignore")
    existing = set()
    if gitignore_path.exists():
        existing = set(gitignore_path.read_text().splitlines())

    for entry in [".mcp.json", ".claude/hooks/.env"]:
        if entry not in existing:
            needed.append(entry)
    return needed


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

    # 1. Get API key
    api_key = _prompt_required("Myelin API key: ")

    # 2. .mcp.json
    mcp_content, mcp_existed = _build_mcp_json(api_key)
    _show_header(".mcp.json")
    _show_content(mcp_content)
    if _confirm("Write .mcp.json?"):
        Path(".mcp.json").write_text(mcp_content)
        verb = "Updated" if mcp_existed else "Created"
        _ok(f"{verb} {BOLD}.mcp.json{RESET}")
    else:
        _skip("Skipped .mcp.json")

    # 3. .claude/hooks/.env
    hooks_dir = Path(".claude/hooks")
    hooks_dir.mkdir(parents=True, exist_ok=True)

    env_content, env_existed = _build_env_content(hooks_dir, api_key)
    _show_header(".claude/hooks/.env")
    _show_content(env_content)
    if _confirm("Write .claude/hooks/.env?"):
        (hooks_dir / ".env").write_text(env_content)
        verb = "Updated" if env_existed else "Created"
        _ok(f"{verb} {BOLD}.claude/hooks/.env{RESET}")
    else:
        _skip("Skipped .claude/hooks/.env")

    # 4. Hook script
    hook_script = hooks_dir / "myelin-capture.py"
    hook_existed = hook_script.exists()
    _show_header(".claude/hooks/myelin-capture.py")
    source = importlib.resources.files("myelin_sdk.claude_code").joinpath("capture.py")
    with importlib.resources.as_file(source) as src_path:
        line_count = len(src_path.read_text().splitlines())
    print(f"  {DIM}│{RESET} (hook script — {line_count} lines)")
    if _confirm("Install capture hook?"):
        _install_hook_script(hooks_dir, "capture.py", "myelin-capture.py")
        verb = "Updated" if hook_existed else "Installed"
        _ok(f"{verb} {BOLD}.claude/hooks/myelin-capture.py{RESET}")
    else:
        _skip("Skipped capture hook")

    # Clean up old hooks if upgrading
    for old_name in ("myelin-capture.sh", "myelin-reasoning.py"):
        old_dest = hooks_dir / old_name
        if old_dest.exists():
            old_dest.unlink()
            _info(f"Removed old {old_name}")

    # 5. .claude/settings.json
    hook_command = 'python3 "$CLAUDE_PROJECT_DIR"/.claude/hooks/myelin-capture.py'
    settings_content, settings_existed, hook_already_present = _build_settings_json(hook_command)

    _show_header(".claude/settings.json")
    if hook_already_present:
        print(f"  {DIM}│{RESET} PostToolUse hook already registered")
        _ok(f"No changes needed for {BOLD}.claude/settings.json{RESET}")
    else:
        print(f"  {DIM}│{RESET} Adding PostToolUse hook:")
        hook_preview = json.dumps({
            "matcher": ".*",
            "hooks": [{
                "type": "command",
                "command": hook_command,
                "async": True,
                "timeout": 10,
            }],
        }, indent=2)
        _show_content(hook_preview)
        if _confirm("Update .claude/settings.json?"):
            settings_path = Path(".claude/settings.json")
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(settings_content)
            _ok(f"Registered PostToolUse hook in {BOLD}.claude/settings.json{RESET}")
        else:
            _skip("Skipped .claude/settings.json")

    # 6. .gitignore
    gitignore_needed = _gitignore_entries_needed()
    if gitignore_needed:
        _show_header(".gitignore")
        print(f"  {DIM}│{RESET} Adding:")
        for entry in gitignore_needed:
            print(f"  {DIM}│{RESET}   {entry}")
        if _confirm("Update .gitignore?"):
            gitignore_path = Path(".gitignore")
            existing = gitignore_path.read_text() if gitignore_path.exists() else ""
            if existing and not existing.endswith("\n"):
                existing += "\n"
            gitignore_path.write_text(existing + "\n".join(gitignore_needed) + "\n")
            _ok(f"Updated {BOLD}.gitignore{RESET}")
        else:
            _skip("Skipped .gitignore")

    # 7. Success summary
    print()
    print(f"  {BOLD_GREEN}Setup complete!{RESET}")
    print()
    print("  Restart Claude Code to activate Myelin. The MCP server")
    print("  and hooks will load automatically from your project config.")
    print()
