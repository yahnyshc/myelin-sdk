"""Tests for reasoning extraction from the JSONL transcript."""

import json
import os
import tempfile

import pytest

from myelin_sdk.claude_code.capture import extract_reasoning_from_transcript


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


class TestTranscriptParsing:
    def test_extracts_thinking_and_text(self, transcript_file):
        """Extracts both thinking and text blocks from the same message."""
        entries = [
            {
                "message": {
                    "id": "msg_001",
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "The user needs a password reset. Let me search first."},
                        {"type": "text", "text": "I'll search for the user by email."},
                        {"type": "tool_use", "id": "tu_abc123", "name": "admin_api.search"},
                    ],
                }
            }
        ]
        path = _write_transcript(transcript_file, entries)

        result = extract_reasoning_from_transcript(path, "tu_abc123")

        assert result is not None
        assert "The user needs a password reset" in result
        assert "I'll search for the user by email" in result
        # Thinking comes before text
        thinking_pos = result.index("password reset")
        text_pos = result.index("search for the user")
        assert thinking_pos < text_pos

    def test_thinking_only(self, transcript_file):
        """Works with only thinking block, no text."""
        entries = [
            {
                "message": {
                    "id": "msg_002",
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Just thinking here"},
                        {"type": "tool_use", "id": "tu_think", "name": "some_tool"},
                    ],
                }
            }
        ]
        path = _write_transcript(transcript_file, entries)

        result = extract_reasoning_from_transcript(path, "tu_think")

        assert result == "Just thinking here"

    def test_text_only(self, transcript_file):
        """Works with only text block, no thinking."""
        entries = [
            {
                "message": {
                    "id": "msg_003",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me do this."},
                        {"type": "tool_use", "id": "tu_text", "name": "some_tool"},
                    ],
                }
            }
        ]
        path = _write_transcript(transcript_file, entries)

        result = extract_reasoning_from_transcript(path, "tu_text")

        assert result == "Let me do this."

    def test_tool_use_id_not_found(self, transcript_file):
        """Returns None when tool_use_id doesn't exist in transcript."""
        entries = [
            {
                "message": {
                    "id": "msg_004",
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu_other", "name": "some_tool"},
                    ],
                }
            }
        ]
        path = _write_transcript(transcript_file, entries)

        result = extract_reasoning_from_transcript(path, "tu_missing")

        assert result is None

    def test_empty_thinking_and_text(self, transcript_file):
        """Returns None when both thinking and text are empty."""
        entries = [
            {
                "message": {
                    "id": "msg_005",
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": ""},
                        {"type": "text", "text": "  "},
                        {"type": "tool_use", "id": "tu_empty", "name": "some_tool"},
                    ],
                }
            }
        ]
        path = _write_transcript(transcript_file, entries)

        result = extract_reasoning_from_transcript(path, "tu_empty")

        assert result is None

    def test_nonexistent_file(self):
        """Returns None for nonexistent transcript file."""
        result = extract_reasoning_from_transcript("/tmp/nonexistent_transcript.jsonl", "tu_x")
        assert result is None

    def test_multiple_messages_finds_correct_one(self, transcript_file):
        """Finds reasoning from the correct message among multiple entries."""
        entries = [
            {
                "message": {
                    "id": "msg_earlier",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Earlier reasoning"},
                        {"type": "tool_use", "id": "tu_earlier", "name": "tool_a"},
                    ],
                }
            },
            {
                "message": {
                    "id": "msg_later",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Later reasoning"},
                        {"type": "tool_use", "id": "tu_later", "name": "tool_b"},
                    ],
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result = extract_reasoning_from_transcript(path, "tu_later")

        assert result == "Later reasoning"

    def test_user_messages_ignored(self, transcript_file):
        """Only parses assistant messages, skips user messages."""
        entries = [
            {
                "message": {
                    "id": "msg_user",
                    "role": "user",
                    "content": [
                        {"type": "tool_use", "id": "tu_user", "name": "fake"},
                    ],
                }
            },
            {
                "message": {
                    "id": "msg_asst",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Real reasoning"},
                        {"type": "tool_use", "id": "tu_real", "name": "real_tool"},
                    ],
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        assert extract_reasoning_from_transcript(path, "tu_user") is None
        assert extract_reasoning_from_transcript(path, "tu_real") == "Real reasoning"

    def test_split_entries_same_message_id(self, transcript_file):
        """Claude Code splits one message across multiple JSONL entries.

        Thinking, text, and tool_use are separate lines sharing the same
        message.id. The parser must collect from all of them.
        """
        entries = [
            {
                "message": {
                    "id": "msg_split",
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me think about this carefully."},
                    ],
                }
            },
            {
                "message": {
                    "id": "msg_split",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll search for the user now."},
                    ],
                }
            },
            {
                "message": {
                    "id": "msg_split",
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu_split", "name": "admin_api.search"},
                    ],
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result = extract_reasoning_from_transcript(path, "tu_split")

        assert result is not None
        assert "Let me think about this carefully" in result
        assert "I'll search for the user now" in result

    def test_split_entries_preserves_order(self, transcript_file):
        """Thinking from earlier entries comes before text from later entries."""
        entries = [
            {
                "message": {
                    "id": "msg_order",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "First text block."},
                    ],
                }
            },
            {
                "message": {
                    "id": "msg_order",
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Deep reasoning here."},
                    ],
                }
            },
            {
                "message": {
                    "id": "msg_order",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Second text block."},
                    ],
                }
            },
            {
                "message": {
                    "id": "msg_order",
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu_order", "name": "some_tool"},
                    ],
                }
            },
        ]
        path = _write_transcript(transcript_file, entries)

        result = extract_reasoning_from_transcript(path, "tu_order")

        assert result is not None
        # Thinking comes before text in output
        assert result.startswith("Deep reasoning here.")
        assert "First text block." in result
        assert "Second text block." in result
