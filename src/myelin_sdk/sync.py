#!/usr/bin/env python3
"""Sync local markdown procedure files to the Myelin server.

Standalone CLI (stdlib-only, no httpx) — uses urllib.request like capture.py.

Usage:
    myelin-sync                           # sync .claude/procedures/*.md
    myelin-sync --dir ./runbooks          # custom directory
    myelin-sync deploy.md hotfix.md       # specific files
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

# -- Defaults ----------------------------------------------------------------

DEFAULT_DIR = os.path.join(".claude", "procedures")
SYNC_TIMEOUT = 30  # seconds — sync may be slower than single captures

# -- Logging -----------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[myelin] {msg}", file=sys.stderr)


def debug(msg: str) -> None:
    if os.environ.get("MYELIN_DEBUG") == "1":
        log(f"DEBUG: {msg}")


# -- Credential loading (reused from capture.py) -----------------------------

_ENV_LOADED = False


def _validate_base_url(url: str) -> str:
    """Validate URL scheme. HTTPS required; HTTP only for localhost."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    localhost = {"localhost", "127.0.0.1", "[::1]", "::1"}

    if scheme == "https":
        return url
    if scheme == "http" and hostname in localhost:
        return url
    if scheme == "http":
        raise ValueError(
            f"Insecure URL: {url!r} — HTTPS required for non-localhost."
        )
    raise ValueError(
        f"Invalid URL scheme: {url!r} — only https:// "
        f"(or http:// for localhost) allowed."
    )


def _load_env() -> None:
    """Derive MYELIN_URL and MYELIN_API_KEY from .mcp.json if not set."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    if os.environ.get("MYELIN_URL") and os.environ.get("MYELIN_API_KEY"):
        return

    # Try CLAUDE_PROJECT_DIR first, then cwd
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())

    mcp_path = os.path.join(project_dir, ".mcp.json")
    try:
        with open(mcp_path) as f:
            config = json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return

    server = config.get("mcpServers", {}).get("myelin")
    if not isinstance(server, dict):
        return

    url = server.get("url", "")
    if url and not os.environ.get("MYELIN_URL"):
        base_url = url.removesuffix("/mcp")
        try:
            _validate_base_url(base_url)
        except ValueError as exc:
            log(f"ignoring URL from .mcp.json: {exc}")
            return
        os.environ["MYELIN_URL"] = base_url

    auth = server.get("headers", {}).get("Authorization", "")
    if auth and not os.environ.get("MYELIN_API_KEY"):
        key = auth.removeprefix("Bearer ").strip()
        if key:
            os.environ["MYELIN_API_KEY"] = key

    if os.environ.get("MYELIN_URL") and os.environ.get("MYELIN_API_KEY"):
        debug(f"loaded credentials from {mcp_path}")


# -- File reading / description extraction -----------------------------------

_HEADING_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)


def _extract_description(content: str, filepath: str) -> str:
    """Extract description from first # heading, fallback to filename."""
    m = _HEADING_RE.search(content)
    if m:
        return m.group(1).strip()
    basename = os.path.splitext(os.path.basename(filepath))[0]
    # Convert kebab/snake case to title
    return basename.replace("-", " ").replace("_", " ").title()


def _resolve_project_root() -> str:
    """Return the project root directory."""
    return os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())


def _relative_path(filepath: str, root: str) -> str:
    """Normalize path relative to project root."""
    abspath = os.path.abspath(filepath)
    try:
        return os.path.relpath(abspath, root)
    except ValueError:
        # Different drives on Windows
        return abspath


def collect_files(
    paths: list[str] | None, directory: str | None
) -> list[dict]:
    """Read markdown files and return list of {path, content, description}.

    Either *paths* (explicit file list) or *directory* (glob *.md) is used.
    """
    root = _resolve_project_root()
    files: list[dict] = []

    if paths:
        targets = paths
    else:
        d = directory or os.path.join(root, DEFAULT_DIR)
        if not os.path.isdir(d):
            log(f"directory not found: {d}")
            return []
        targets = sorted(
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.endswith(".md")
        )

    if not targets:
        log("no markdown files found")
        return []

    for filepath in targets:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except (OSError, IOError) as exc:
            log(f"skipping {filepath}: {exc}")
            continue

        if not content.strip():
            log(f"skipping empty file: {filepath}")
            continue

        rel_path = _relative_path(filepath, root)
        description = _extract_description(content, filepath)
        files.append({
            "path": rel_path,
            "content": content,
            "description": description,
        })

    return files


# -- API call ----------------------------------------------------------------


def sync_to_server(files: list[dict]) -> dict | None:
    """POST files to /v1/workflows/sync. Returns response JSON or None."""
    myelin_url = os.environ.get("MYELIN_URL", "")
    myelin_key = os.environ.get("MYELIN_API_KEY", "")

    if not myelin_url or not myelin_key:
        log(
            "MYELIN_URL and MYELIN_API_KEY must be set. "
            "Run: export MYELIN_URL=... MYELIN_API_KEY=... "
            "or configure .mcp.json"
        )
        return None

    try:
        _validate_base_url(myelin_url)
    except ValueError as exc:
        log(str(exc))
        return None

    payload = json.dumps({"files": files}).encode()

    req = urllib.request.Request(
        f"{myelin_url}/v1/workflows/sync",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {myelin_key}",
        },
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=SYNC_TIMEOUT)
        return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode()
        except Exception:
            pass
        log(f"server returned {exc.code}: {body}")
        return None
    except (urllib.error.URLError, OSError) as exc:
        log(f"request failed: {exc}")
        return None


# -- CLI entry point ---------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="myelin-sync",
        description="Sync local markdown procedures to the Myelin server.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific .md files to sync (default: all in --dir)",
    )
    parser.add_argument(
        "--dir",
        dest="directory",
        default=None,
        help=(
            f"Directory containing .md files "
            f"(default: {DEFAULT_DIR})"
        ),
    )
    args = parser.parse_args()

    _load_env()

    paths = args.files if args.files else None
    files = collect_files(paths, args.directory)

    if not files:
        print("No procedures to sync.")
        return 1

    debug(f"syncing {len(files)} file(s)")

    result = sync_to_server(files)
    if result is None:
        return 1

    created = result.get("created", 0)
    updated = result.get("updated", 0)
    unchanged = result.get("unchanged", 0)
    total = created + updated + unchanged

    print(
        f"Synced {total} procedures "
        f"({created} new, {updated} updated, {unchanged} unchanged)"
    )

    # Show per-file details when there are few files
    details = result.get("details", [])
    if details and len(details) <= 20:
        for entry in details:
            status = entry.get("status", "unknown")
            path = entry.get("path", "?")
            marker = {"created": "+", "updated": "~", "unchanged": "="}.get(
                status, "?"
            )
            print(f"  {marker} {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
