"""Shared utilities for Myelin SDK internals."""

MAX_RESPONSE_LEN = 8000
_HEAD = MAX_RESPONSE_LEN // 2
_TAIL = MAX_RESPONSE_LEN // 2


def truncate(text: str) -> str:
    """Keep first and last chars so the evaluator sees both the start and outcome."""
    if len(text) <= MAX_RESPONSE_LEN:
        return text
    return (
        text[:_HEAD]
        + f"\n... [{len(text)} chars, middle truncated] ...\n"
        + text[-_TAIL:]
    )
