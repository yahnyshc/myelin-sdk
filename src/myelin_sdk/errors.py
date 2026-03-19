"""Human-readable API errors for Myelin SDK."""

from __future__ import annotations

import httpx

_HINTS: dict[int, str] = {
    401: "Check your API key (MYELIN_API_KEY or api_key= param).",
    404: "Session not found — it may have expired.",
    413: "Payload too large.",
    429: "Rate limit exceeded.",
}


class MyelinAPIError(httpx.HTTPStatusError):
    """Wraps httpx.HTTPStatusError with a human-readable message.

    Existing ``except httpx.HTTPStatusError`` clauses still catch this.
    """

    def __init__(self, *, response: httpx.Response, request: httpx.Request) -> None:
        self.status_code = response.status_code

        # Try to extract server error message
        server_msg = ""
        try:
            body = response.json()
            server_msg = body.get("error", "")
        except Exception:
            server_msg = response.reason_phrase or ""

        if not server_msg:
            server_msg = response.reason_phrase or "Unknown error"

        hint = _HINTS.get(self.status_code, "")

        # Include Retry-After for 429
        if self.status_code == 429:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                hint = f"Rate limit exceeded. Retry after {retry_after}s."

        parts = [f"Myelin API error ({self.status_code}): {server_msg}"]
        if hint:
            parts.append(hint)
        self.detail = " ".join(parts)

        # Strip auth headers to prevent API key leaks in tracebacks
        safe_request = httpx.Request(request.method, request.url)
        super().__init__(self.detail, request=safe_request, response=response)


def raise_for_status(response: httpx.Response) -> None:
    """Drop-in replacement for ``response.raise_for_status()``."""
    if response.is_success:
        return
    raise MyelinAPIError(response=response, request=response.request)
