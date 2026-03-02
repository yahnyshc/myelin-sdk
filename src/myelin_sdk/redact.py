"""Redaction engine for Myelin.

Scrubs sensitive data (API keys, tokens, passwords, credentials) from tool call
data before it leaves the machine. Two layers:

1. Key-name denylist — scrub dict values where key matches sensitive names
2. Regex value scanning — detect secrets by pattern (AWS, GitHub, Stripe, etc.)

Used by both the Claude Code capture hook and the LangChain callback handler.
"""

import json
import os
import re

# -- Built-in patterns --------------------------------------------------------

BUILTIN_PATTERNS: list[dict] = [
    # Cloud — AWS
    {
        "name": "aws_access_key",
        "pattern": r"(?<![A-Z0-9])(AKIA[0-9A-Z]{16})(?![A-Z0-9])",
        "description": "AWS access key ID",
    },
    {
        "name": "aws_secret_key",
        "pattern": r"(?<![A-Za-z0-9/+=])([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])",
        "description": "AWS secret access key (40-char base64)",
        "_context_required": True,
    },
    # Git — GitHub
    {
        "name": "github_pat",
        "pattern": r"\bghp_[A-Za-z0-9]{36,}\b",
        "description": "GitHub personal access token",
    },
    {
        "name": "github_fine_grained_pat",
        "pattern": r"\bgithub_pat_[A-Za-z0-9_]{22,}\b",
        "description": "GitHub fine-grained personal access token",
    },
    {
        "name": "github_oauth",
        "pattern": r"\bgho_[A-Za-z0-9]{36,}\b",
        "description": "GitHub OAuth access token",
    },
    {
        "name": "github_user_token",
        "pattern": r"\bghu_[A-Za-z0-9]{36,}\b",
        "description": "GitHub user-to-server token",
    },
    {
        "name": "github_server_token",
        "pattern": r"\bghs_[A-Za-z0-9]{36,}\b",
        "description": "GitHub server-to-server token",
    },
    # Auth — Bearer / Basic
    {
        "name": "bearer_token",
        "pattern": r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b",
        "description": "Bearer authentication token",
    },
    {
        "name": "basic_auth",
        "pattern": r"(?i)\bBasic\s+[A-Za-z0-9+/]+=*\b",
        "description": "Basic authentication header",
    },
    # AI providers
    {
        "name": "anthropic_api_key",
        "pattern": r"\bsk-ant-[A-Za-z0-9\-]{20,}\b",
        "description": "Anthropic API key",
    },
    {
        "name": "openai_api_key",
        "pattern": r"\bsk-[A-Za-z0-9]{20,}T3BlbkFJ[A-Za-z0-9]{20,}\b",
        "description": "OpenAI API key",
    },
    # Payment — Stripe
    {
        "name": "stripe_secret_key",
        "pattern": r"\bsk_(live|test)_[A-Za-z0-9]{24,}\b",
        "description": "Stripe secret key",
    },
    {
        "name": "stripe_restricted_key",
        "pattern": r"\brk_(live|test)_[A-Za-z0-9]{24,}\b",
        "description": "Stripe restricted key",
    },
    # Infra — database URLs with credentials
    {
        "name": "database_url",
        "pattern": r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s:]+:[^\s@]+@[^\s]+",
        "description": "Database connection URL with credentials",
    },
    # Infra — PEM private keys
    {
        "name": "private_key",
        "pattern": r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        "description": "PEM private key header",
    },
    # Services — Slack
    {
        "name": "slack_token",
        "pattern": r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b",
        "description": "Slack API token",
    },
    # Services — SendGrid
    {
        "name": "sendgrid_key",
        "pattern": r"\bSG\.[A-Za-z0-9\-_]{22,}\.[A-Za-z0-9\-_]{22,}\b",
        "description": "SendGrid API key",
    },
    # Services — Google
    {
        "name": "google_api_key",
        "pattern": r"\bAIza[A-Za-z0-9\-_]{35}\b",
        "description": "Google API key",
    },
    # Services — Twilio
    {
        "name": "twilio_api_key",
        "pattern": r"\bSK[a-f0-9]{32}\b",
        "description": "Twilio API key",
    },
    # Services — npm
    {
        "name": "npm_token",
        "pattern": r"\bnpm_[A-Za-z0-9]{36,}\b",
        "description": "npm access token",
    },
    # Services — PyPI
    {
        "name": "pypi_token",
        "pattern": r"\bpypi-[A-Za-z0-9\-_]{50,}\b",
        "description": "PyPI API token",
    },
    # Generic — JWT
    {
        "name": "jwt",
        "pattern": r"\beyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_.+/=]+\b",
        "description": "JSON Web Token",
    },
    # Generic — password in URL
    {
        "name": "password_in_url",
        "pattern": r"://[^\s:]+:[^\s@]+@",
        "description": "Password embedded in URL",
    },
]

# Key names whose dict values should always be scrubbed (case-insensitive)
_DEFAULT_SENSITIVE_KEYS: frozenset[str] = frozenset({
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "api-key",
    "authorization",
    "credentials",
    "access_token",
    "refresh_token",
    "private_key",
    "secret_key",
    "client_secret",
})


# -- Compiled pattern cache ---------------------------------------------------

def _compile_patterns(
    patterns: list[dict],
) -> list[tuple[str, re.Pattern]]:
    """Compile pattern dicts into (name, regex) tuples."""
    compiled = []
    for p in patterns:
        if p.get("_context_required"):
            continue  # skip context-dependent patterns in generic scanning
        compiled.append((p["name"], re.compile(p["pattern"])))
    return compiled


# -- Config class --------------------------------------------------------------


class RedactionConfig:
    """Configuration for what and how to redact."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        additional_patterns: list[dict] | None = None,
        additional_keys: list[str] | None = None,
        patterns: list[dict] | None = None,
        sensitive_keys: list[str] | None = None,
        replacement: str = "[REDACTED]",
        redact_tool_input: bool = True,
        redact_tool_response: bool = True,
        redact_reasoning: bool = True,
    ):
        self.enabled = enabled
        self.replacement = replacement
        self.redact_tool_input = redact_tool_input
        self.redact_tool_response = redact_tool_response
        self.redact_reasoning = redact_reasoning

        # Build sensitive keys set
        if sensitive_keys is not None:
            self._sensitive_keys = {k.lower() for k in sensitive_keys}
        else:
            self._sensitive_keys = set(_DEFAULT_SENSITIVE_KEYS)
        if additional_keys:
            self._sensitive_keys |= {k.lower() for k in additional_keys}

        # Build pattern list
        if patterns is not None:
            raw_patterns = list(patterns)
        else:
            raw_patterns = list(BUILTIN_PATTERNS)
            if additional_patterns:
                raw_patterns.extend(additional_patterns)

        self._patterns = raw_patterns
        self._compiled = _compile_patterns(raw_patterns)

    @classmethod
    def from_dict(cls, data: dict) -> "RedactionConfig":
        """Create config from a parsed dict (e.g. JSON file content)."""
        return cls(
            enabled=data.get("enabled", True),
            additional_patterns=data.get("additional_patterns"),
            additional_keys=data.get("additional_keys"),
            patterns=data.get("patterns"),
            sensitive_keys=data.get("sensitive_keys"),
            replacement=data.get("replacement", "[REDACTED]"),
            redact_tool_input=data.get("redact_tool_input", True),
            redact_tool_response=data.get("redact_tool_response", True),
            redact_reasoning=data.get("redact_reasoning", True),
        )

    @classmethod
    def from_file(cls, path: str) -> "RedactionConfig":
        """Load config from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_env(cls) -> "RedactionConfig":
        """Resolve config from environment variables and auto-discovery.

        Resolution order:
        1. MYELIN_REDACT=0 → disabled
        2. MYELIN_REDACTION_CONFIG=/path → load from explicit path
        3. Auto-discover $CLAUDE_PROJECT_DIR/.claude/hooks/redaction.json
        4. Fall back to built-in defaults
        """
        if os.environ.get("MYELIN_REDACT") == "0":
            return cls(enabled=False)

        config_path = os.environ.get("MYELIN_REDACTION_CONFIG", "")
        if config_path and os.path.isfile(config_path):
            return cls.from_file(config_path)

        project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
        if project_dir:
            auto_path = os.path.join(
                project_dir, ".claude", "hooks", "redaction.json"
            )
            if os.path.isfile(auto_path):
                return cls.from_file(auto_path)

        return cls()


# -- Redaction functions -------------------------------------------------------


def redact_string(text: str, config: "RedactionConfig | None" = None) -> str:
    """Apply regex patterns to scrub secrets from a string."""
    if config is None:
        config = _get_default()
    if not config.enabled or not text:
        return text
    for _name, pattern in config._compiled:
        text = pattern.sub(config.replacement, text)
    return text


def redact_dict(
    data: dict, config: "RedactionConfig | None" = None
) -> dict:
    """Scrub sensitive dict values by key name, then regex-scan all string values."""
    if config is None:
        config = _get_default()
    if not config.enabled:
        return data
    return _redact_value(data, config)


def _redact_value(value, config: "RedactionConfig"):
    """Recursively redact a value (dict, list, or string)."""
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if k.lower() in config._sensitive_keys and isinstance(v, str):
                result[k] = config.replacement
            else:
                result[k] = _redact_value(v, config)
        return result
    if isinstance(value, list):
        return [_redact_value(item, config) for item in value]
    if isinstance(value, str):
        return redact_string(value, config)
    return value


# -- Default config singleton --------------------------------------------------

_default_config: "RedactionConfig | None" = None


def get_default_config() -> RedactionConfig:
    """Return a lazily-initialized default RedactionConfig."""
    global _default_config
    if _default_config is None:
        _default_config = RedactionConfig()
    return _default_config


def _get_default() -> RedactionConfig:
    """Internal alias for get_default_config."""
    return get_default_config()


def build_default_redaction_dict() -> dict:
    """Generate the full default redaction config as a plain dict.

    Suitable for writing to ``redaction.json``. The returned dict contains
    all built-in patterns (with internal keys like ``_context_required``
    stripped) and all default sensitive key names.
    """
    patterns = []
    for p in BUILTIN_PATTERNS:
        clean = {k: v for k, v in p.items() if not k.startswith("_")}
        patterns.append(clean)

    return {
        "enabled": True,
        "replacement": "[REDACTED]",
        "redact_tool_input": True,
        "redact_tool_response": True,
        "redact_reasoning": True,
        "patterns": patterns,
        "sensitive_keys": sorted(_DEFAULT_SENSITIVE_KEYS),
    }
