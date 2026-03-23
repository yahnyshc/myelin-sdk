"""Tests for context extraction from the JSONL transcript."""

import json
import os
import tempfile

import pytest

from myelin_sdk.claude_code.capture import (
    MAX_CONTEXT_LEN,
    extract_context_from_transcript,
)


@pytest.fixture
def transcript_file():
    """Create a temp JSONL transcript file and clean up after."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    yield f
    f.close()
    os.unlink(f.name)


def _write_transcript(f, entries: list[dict]) -> str:
    """Write JSONL entries to file and return path."""
    for entry in entries:
        f.write(json.dumps(entry) + "\n")
    f.flush()
    return f.name


class TestContextExtraction:
    def test_basic_extraction(self, transcript_file):
        """Thinking + text + user messages produce correct tagged output."""
        entries = [
            {
                "message": {
                    "id": "msg_001",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "The user needs a password reset.",
                        },
                        {
                            "type": "text",
                            "text": "I'll search for the user by email.",
                        },
                        {
                            "type": "tool_use",
                            "id": "tu_abc",
                            "name": "search",
                        },
                    ],
                }
            },
            {
                "message": {
                    "id": "msg_002",
                    "role": "user",
                    "content": "Great, thanks for looking.",
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result, new_offset = extract_context_from_transcript(path, 0)

        assert result is not None
        assert "[thinking] The user needs a password reset." in result
        assert "[assistant] I'll search for the user by email." in result
        assert "[user] Great, thanks for looking." in result
        assert new_offset == 2

    def test_offset_skips_pre_session(self, transcript_file):
        """Lines before offset are not included in context."""
        entries = [
            {
                "message": {
                    "id": "msg_pre",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Before session."},
                    ],
                }
            },
            {
                "message": {
                    "id": "msg_post",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "After session start."},
                    ],
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result, new_offset = extract_context_from_transcript(path, 1)

        assert result is not None
        assert "Before session" not in result
        assert "[assistant] After session start." in result
        assert new_offset == 2

    def test_tool_use_filtered(self, transcript_file):
        """tool_use blocks are not included in context output."""
        entries = [
            {
                "message": {
                    "id": "msg_tu",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Searching now."},
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "search",
                        },
                    ],
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result, _ = extract_context_from_transcript(path, 0)

        assert result is not None
        assert "[assistant] Searching now." in result
        assert "tool_use" not in result
        assert "tu_1" not in result

    def test_tool_result_filtered(self, transcript_file):
        """User messages with tool_result content are skipped entirely."""
        entries = [
            {
                "message": {
                    "id": "msg_tr",
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": "file contents here",
                        },
                    ],
                }
            },
            {
                "message": {
                    "id": "msg_real",
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Can you fix it?"},
                    ],
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result, _ = extract_context_from_transcript(path, 0)

        assert result is not None
        assert "file contents here" not in result
        assert "tool_result" not in result
        assert "[user] Can you fix it?" in result

    def test_user_string_content(self, transcript_file):
        """User messages with plain string content are captured as [user]."""
        entries = [
            {
                "message": {
                    "id": "msg_str",
                    "role": "user",
                    "content": "Please help me deploy.",
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result, _ = extract_context_from_transcript(path, 0)

        assert result == "[user] Please help me deploy."

    def test_user_array_text_content(self, transcript_file):
        """User messages with text block array are captured."""
        entries = [
            {
                "message": {
                    "id": "msg_arr",
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First part."},
                        {"type": "text", "text": "Second part."},
                    ],
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result, _ = extract_context_from_transcript(path, 0)

        assert result is not None
        assert "[user] First part." in result
        assert "[user] Second part." in result

    def test_progress_skipped(self, transcript_file):
        """Lines without a message field (e.g. progress entries) are skipped."""
        entries = [
            {"type": "progress", "data": "50%"},
            {
                "message": {
                    "id": "msg_real",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Done."},
                    ],
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result, _ = extract_context_from_transcript(path, 0)

        assert result == "[assistant] Done."

    def test_file_history_skipped(self, transcript_file):
        """Lines with type file-history-snapshot (no message) are skipped."""
        entries = [
            {"type": "file-history-snapshot", "files": ["/a.py"]},
            {
                "message": {
                    "id": "msg_ok",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "All good."},
                    ],
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result, _ = extract_context_from_transcript(path, 0)

        assert result == "[assistant] All good."

    def test_empty_returns_none(self, transcript_file):
        """Empty transcript returns (None, offset)."""
        path = _write_transcript(transcript_file, [])

        result, new_offset = extract_context_from_transcript(path, 0)

        assert result is None
        assert new_offset == 0

    def test_missing_file(self):
        """Missing transcript path returns (None, offset)."""
        result, new_offset = extract_context_from_transcript(
            "/tmp/nonexistent_transcript.jsonl", 5
        )

        assert result is None
        assert new_offset == 5

    def test_truncation(self, transcript_file):
        """Text over MAX_CONTEXT_LEN gets head+tail truncated."""
        # Create a large message that exceeds MAX_CONTEXT_LEN
        big_text = "x" * (MAX_CONTEXT_LEN + 10000)
        entries = [
            {
                "message": {
                    "id": "msg_big",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": big_text},
                    ],
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result, _ = extract_context_from_transcript(path, 0)

        assert result is not None
        assert "middle truncated" in result
        assert len(result) > MAX_CONTEXT_LEN  # includes truncation marker

    def test_returns_new_offset(self, transcript_file):
        """Returned offset matches total line count of transcript."""
        entries = [
            {
                "message": {
                    "id": f"msg_{i}",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"Line {i}"},
                    ],
                }
            }
            for i in range(5)
        ]
        path = _write_transcript(transcript_file, entries)

        _, new_offset = extract_context_from_transcript(path, 0)

        assert new_offset == 5

    def test_offset_at_end_returns_none(self, transcript_file):
        """When offset >= line count, returns (None, new_offset)."""
        entries = [
            {
                "message": {
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hello"},
                    ],
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result, new_offset = extract_context_from_transcript(path, 1)

        assert result is None
        assert new_offset == 1

    def test_system_role_skipped(self, transcript_file):
        """Messages with role 'system' are not included."""
        entries = [
            {
                "message": {
                    "id": "msg_sys",
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "System prompt"},
                    ],
                }
            },
            {
                "message": {
                    "id": "msg_asst",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hello!"},
                    ],
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result, _ = extract_context_from_transcript(path, 0)

        assert result == "[assistant] Hello!"
        assert "System prompt" not in result
