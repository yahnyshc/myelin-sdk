"""Shared utilities for Myelin SDK internals."""

from urllib.parse import urlparse

MAX_RESPONSE_LEN = 8000
MAX_CONTEXT_LEN = 100_000

_LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}


def validate_base_url(url: str) -> str:
    """Validate that a Myelin base URL uses HTTPS (or HTTP for localhost only).

    Returns the URL unchanged if valid. Raises ValueError if the scheme
    is not HTTPS or the host is not localhost when using HTTP.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()

    if scheme == "https":
        return url
    if scheme == "http" and hostname in _LOCALHOST_HOSTS:
        return url

    if scheme == "http":
        raise ValueError(
            f"Insecure base_url: {url!r} — HTTPS is required for non-localhost URLs. "
            f"This prevents API keys from being sent to unintended servers."
        )
    raise ValueError(
        f"Invalid base_url scheme: {url!r} — only https:// (or http:// for localhost) is allowed."
    )
def truncate(text: str, max_len: int = MAX_RESPONSE_LEN) -> str:
    """Keep first and last chars so the evaluator sees both the start and outcome."""
    if len(text) <= max_len:
        return text
    head = max_len // 2
    tail = max_len // 2
    return (
        text[:head]
        + f"\n... [{len(text)} chars, middle truncated] ...\n"
        + text[-tail:]
    )
