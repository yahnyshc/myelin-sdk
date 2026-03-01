"""Tests for the redaction engine."""

import json
import os
import tempfile

from myelin_sdk.redact import (
    BUILTIN_PATTERNS,
    RedactionConfig,
    build_default_redaction_dict,
    get_default_config,
    redact_dict,
    redact_string,
)


# -- Pattern detection tests ---------------------------------------------------


class TestPatternDetection:
    """One test per built-in pattern to verify it detects real secrets."""

    def test_aws_access_key(self):
        text = "key is AKIAIOSFODNN7EXAMPLE"
        assert "[REDACTED]" in redact_string(text)

    def test_github_pat(self):
        text = "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        assert "[REDACTED]" in redact_string(text)

    def test_github_fine_grained_pat(self):
        text = "github_pat_11AAAAAA0abcdefghijklmnopqrst"
        assert "[REDACTED]" in redact_string(text)

    def test_github_oauth(self):
        text = "gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        assert "[REDACTED]" in redact_string(text)

    def test_github_user_token(self):
        text = "ghu_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        assert "[REDACTED]" in redact_string(text)

    def test_github_server_token(self):
        text = "ghs_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        assert "[REDACTED]" in redact_string(text)

    def test_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.test"
        result = redact_string(text)
        assert "Bearer" not in result or "[REDACTED]" in result

    def test_basic_auth(self):
        text = "Authorization: Basic dXNlcjpwYXNz"
        result = redact_string(text)
        assert "dXNlcjpwYXNz" not in result

    def test_anthropic_api_key(self):
        text = "sk-ant-api03-abcdefghijklmnopqrst"
        assert "[REDACTED]" in redact_string(text)

    def test_openai_api_key(self):
        text = "sk-abcdefghijklmnopqrstT3BlbkFJabcdefghijklmnopqrst"
        assert "[REDACTED]" in redact_string(text)

    def test_stripe_secret_key(self):
        text = "sk_live_abcdefghijklmnopqrstuvwx"
        assert "[REDACTED]" in redact_string(text)

    def test_stripe_test_key(self):
        text = "sk_test_abcdefghijklmnopqrstuvwx"
        assert "[REDACTED]" in redact_string(text)

    def test_stripe_restricted_key(self):
        text = "rk_live_abcdefghijklmnopqrstuvwx"
        assert "[REDACTED]" in redact_string(text)

    def test_database_url_postgres(self):
        text = "postgresql://admin:s3cret@db.host.com:5432/mydb"
        result = redact_string(text)
        assert "s3cret" not in result

    def test_database_url_mysql(self):
        text = "mysql://root:password123@localhost/app"
        result = redact_string(text)
        assert "password123" not in result

    def test_database_url_mongodb(self):
        text = "mongodb+srv://user:pass@cluster.mongodb.net/db"
        result = redact_string(text)
        assert "pass" not in result or "[REDACTED]" in result

    def test_private_key(self):
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."
        assert "[REDACTED]" in redact_string(text)

    def test_slack_token(self):
        text = "xoxb-1234567890-abcdefghij"
        assert "[REDACTED]" in redact_string(text)

    def test_sendgrid_key(self):
        text = "SG.abcdefghijklmnopqrstuv.abcdefghijklmnopqrstuv"
        assert "[REDACTED]" in redact_string(text)

    def test_google_api_key(self):
        text = "AIzaSyA-abcdefghijklmnopqrstuvwxyz12345"
        assert "[REDACTED]" in redact_string(text)

    def test_twilio_api_key(self):
        text = "SK0123456789abcdef0123456789abcdef"
        assert "[REDACTED]" in redact_string(text)

    def test_npm_token(self):
        text = "npm_abcdefghijklmnopqrstuvwxyz0123456789AB"
        assert "[REDACTED]" in redact_string(text)

    def test_pypi_token(self):
        token = "pypi-" + "A" * 50
        assert "[REDACTED]" in redact_string(token)

    def test_jwt(self):
        text = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        assert "[REDACTED]" in redact_string(text)

    def test_password_in_url(self):
        text = "https://user:secret@example.com/api"
        result = redact_string(text)
        assert "secret" not in result


# -- False positive tests ------------------------------------------------------


class TestFalsePositives:
    """Ensure normal text is NOT redacted."""

    def test_normal_text(self):
        text = "Hello, this is a normal message about Python programming."
        assert redact_string(text) == text

    def test_short_strings(self):
        text = "ok"
        assert redact_string(text) == text

    def test_non_secret_url(self):
        text = "https://example.com/api/v1/users"
        assert redact_string(text) == text

    def test_hex_string_not_twilio(self):
        # 32-char hex that doesn't start with SK
        text = "ab0123456789abcdef0123456789abcdef"
        assert redact_string(text) == text

    def test_normal_base64(self):
        # Short base64 that shouldn't trigger
        text = "data: aGVsbG8="
        assert redact_string(text) == text

    def test_code_snippet(self):
        text = "def calculate(x, y): return x + y"
        assert redact_string(text) == text


# -- Key-name denylist tests ---------------------------------------------------


class TestKeyDenylist:
    """Test dict key-name scrubbing."""

    def test_password_key_redacted(self):
        data = {"username": "admin", "password": "s3cret"}
        result = redact_dict(data)
        assert result["username"] == "admin"
        assert result["password"] == "[REDACTED]"

    def test_api_key_redacted(self):
        data = {"api_key": "sk-abc123", "model": "gpt-4"}
        result = redact_dict(data)
        assert result["api_key"] == "[REDACTED]"
        assert result["model"] == "gpt-4"

    def test_case_insensitive(self):
        data = {"Authorization": "Bearer xyz", "TOKEN": "abc"}
        result = redact_dict(data)
        assert result["Authorization"] == "[REDACTED]"
        assert result["TOKEN"] == "[REDACTED]"

    def test_nested_dict(self):
        data = {
            "config": {
                "database": {
                    "password": "db_pass",
                    "host": "localhost",
                }
            }
        }
        result = redact_dict(data)
        assert result["config"]["database"]["password"] == "[REDACTED]"
        assert result["config"]["database"]["host"] == "localhost"

    def test_list_in_dict(self):
        data = {
            "items": [
                {"token": "secret1", "name": "a"},
                {"token": "secret2", "name": "b"},
            ]
        }
        result = redact_dict(data)
        assert result["items"][0]["token"] == "[REDACTED]"
        assert result["items"][1]["token"] == "[REDACTED]"
        assert result["items"][0]["name"] == "a"

    def test_non_string_values_untouched(self):
        data = {"password": 12345, "secret": True}
        result = redact_dict(data)
        # Only string values are redacted by key name
        assert result["password"] == 12345
        assert result["secret"] is True

    def test_regex_applied_to_string_values(self):
        data = {"note": "Use ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn to auth"}
        result = redact_dict(data)
        assert "ghp_" not in result["note"]
        assert "[REDACTED]" in result["note"]


# -- Config tests --------------------------------------------------------------


class TestConfigFromDict:
    def test_basic_from_dict(self):
        cfg = RedactionConfig.from_dict({"enabled": True, "replacement": "***"})
        assert cfg.enabled is True
        assert cfg.replacement == "***"

    def test_disabled(self):
        cfg = RedactionConfig.from_dict({"enabled": False})
        assert cfg.enabled is False
        text = "sk-ant-api03-secretsecretsecretsecret"
        assert redact_string(text, cfg) == text

    def test_field_toggles(self):
        cfg = RedactionConfig.from_dict({
            "redact_tool_input": False,
            "redact_tool_response": True,
            "redact_reasoning": False,
        })
        assert cfg.redact_tool_input is False
        assert cfg.redact_tool_response is True
        assert cfg.redact_reasoning is False


class TestConfigFromFile:
    def test_loads_json_file(self):
        config_data = {
            "enabled": True,
            "replacement": "***",
            "remove_patterns": ["jwt"],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            f.flush()
            path = f.name

        try:
            cfg = RedactionConfig.from_file(path)
            assert cfg.enabled is True
            assert cfg.replacement == "***"
            # JWT should be removed
            pattern_names = [n for n, _ in cfg._compiled]
            assert "jwt" not in pattern_names
        finally:
            os.unlink(path)


class TestConfigFromEnv:
    def test_myelin_redact_disabled(self, monkeypatch):
        monkeypatch.setenv("MYELIN_REDACT", "0")
        cfg = RedactionConfig.from_env()
        assert cfg.enabled is False

    def test_myelin_redact_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("MYELIN_REDACT", raising=False)
        monkeypatch.delenv("MYELIN_REDACTION_CONFIG", raising=False)
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        cfg = RedactionConfig.from_env()
        assert cfg.enabled is True

    def test_explicit_config_path(self, monkeypatch):
        config_data = {"enabled": True, "replacement": "XXX"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.delenv("MYELIN_REDACT", raising=False)
            monkeypatch.setenv("MYELIN_REDACTION_CONFIG", path)
            cfg = RedactionConfig.from_env()
            assert cfg.replacement == "XXX"
        finally:
            os.unlink(path)

    def test_auto_discover_project_dir(self, monkeypatch, tmp_path):
        hooks_dir = tmp_path / ".claude" / "hooks"
        hooks_dir.mkdir(parents=True)
        config_path = hooks_dir / "redaction.json"
        config_path.write_text(json.dumps({"replacement": "AUTO"}))

        monkeypatch.delenv("MYELIN_REDACT", raising=False)
        monkeypatch.delenv("MYELIN_REDACTION_CONFIG", raising=False)
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
        cfg = RedactionConfig.from_env()
        assert cfg.replacement == "AUTO"


# -- Pattern customization tests -----------------------------------------------


class TestPatternCustomization:
    def test_extra_patterns(self):
        cfg = RedactionConfig(
            extra_patterns=[
                {"name": "custom", "pattern": r"myco_[A-Za-z0-9]{16}"},
            ]
        )
        text = "key is myco_ABCDEFGH12345678"
        assert "[REDACTED]" in redact_string(text, cfg)

    def test_remove_patterns(self):
        cfg = RedactionConfig(remove_patterns=["jwt"])
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        # JWT should NOT be redacted since we removed the pattern
        assert redact_string(jwt, cfg) == jwt

    def test_full_override_patterns(self):
        cfg = RedactionConfig(
            patterns=[
                {"name": "only_this", "pattern": r"SECRET_\d{4}"},
            ]
        )
        # Custom pattern matches
        assert "[REDACTED]" in redact_string("SECRET_1234", cfg)
        # Built-in patterns are gone
        assert redact_string("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn", cfg) == \
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"

    def test_extra_sensitive_keys(self):
        cfg = RedactionConfig(extra_sensitive_keys=["x-custom-auth"])
        data = {"x-custom-auth": "my-secret", "name": "test"}
        result = redact_dict(data, cfg)
        assert result["x-custom-auth"] == "[REDACTED]"
        assert result["name"] == "test"

    def test_custom_replacement(self):
        cfg = RedactionConfig(replacement="***")
        text = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        assert "***" in redact_string(text, cfg)
        assert "[REDACTED]" not in redact_string(text, cfg)


# -- Edge cases ----------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string(self):
        assert redact_string("") == ""

    def test_none_in_dict(self):
        data = {"key": None, "password": "secret"}
        result = redact_dict(data)
        assert result["key"] is None
        assert result["password"] == "[REDACTED]"

    def test_deeply_nested(self):
        data = {
            "a": {"b": {"c": {"d": {"secret": "deep_value"}}}}
        }
        result = redact_dict(data)
        assert result["a"]["b"]["c"]["d"]["secret"] == "[REDACTED]"

    def test_empty_dict(self):
        assert redact_dict({}) == {}

    def test_disabled_config_passthrough(self):
        cfg = RedactionConfig(enabled=False)
        text = "sk-ant-api03-secretsecretsecretsecret"
        assert redact_string(text, cfg) == text
        data = {"password": "secret"}
        assert redact_dict(data, cfg) == data

    def test_mixed_types_in_list(self):
        data = {"items": ["text", 42, True, None, {"token": "abc"}]}
        result = redact_dict(data)
        assert result["items"][0] == "text"
        assert result["items"][1] == 42
        assert result["items"][2] is True
        assert result["items"][3] is None
        assert result["items"][4]["token"] == "[REDACTED]"


# -- get_default_config --------------------------------------------------------


class TestDefaultConfig:
    def test_returns_config(self):
        cfg = get_default_config()
        assert isinstance(cfg, RedactionConfig)
        assert cfg.enabled is True

    def test_singleton(self):
        cfg1 = get_default_config()
        cfg2 = get_default_config()
        assert cfg1 is cfg2


# -- BUILTIN_PATTERNS export ---------------------------------------------------


class TestBuiltinPatterns:
    def test_is_list(self):
        assert isinstance(BUILTIN_PATTERNS, list)
        assert len(BUILTIN_PATTERNS) > 15

    def test_all_have_required_fields(self):
        for p in BUILTIN_PATTERNS:
            assert "name" in p
            assert "pattern" in p
            assert "description" in p


# -- sensitive_keys override tests ---------------------------------------------


class TestSensitiveKeysOverride:
    """Test the sensitive_keys parameter on RedactionConfig."""

    def test_full_override_replaces_defaults(self):
        cfg = RedactionConfig(sensitive_keys=["custom_key"])
        assert cfg._sensitive_keys == {"custom_key"}
        # Default keys should NOT be present
        data = {"password": "secret", "custom_key": "val"}
        result = redact_dict(data, cfg)
        assert result["password"] == "secret"
        assert result["custom_key"] == "[REDACTED]"

    def test_combines_with_extra(self):
        cfg = RedactionConfig(
            sensitive_keys=["custom_key"],
            extra_sensitive_keys=["bonus"],
        )
        assert cfg._sensitive_keys == {"custom_key", "bonus"}

    def test_none_uses_defaults(self):
        cfg = RedactionConfig(sensitive_keys=None)
        data = {"password": "secret"}
        result = redact_dict(data, cfg)
        assert result["password"] == "[REDACTED]"

    def test_from_dict_roundtrip(self):
        cfg = RedactionConfig.from_dict({
            "sensitive_keys": ["my_token", "my_secret"],
        })
        assert cfg._sensitive_keys == {"my_token", "my_secret"}
        data = {"my_token": "abc", "password": "xyz"}
        result = redact_dict(data, cfg)
        assert result["my_token"] == "[REDACTED]"
        assert result["password"] == "xyz"


# -- build_default_redaction_dict tests ----------------------------------------


class TestBuildDefaultRedactionDict:
    """Test the build_default_redaction_dict() helper."""

    def test_has_all_fields(self):
        d = build_default_redaction_dict()
        for key in (
            "enabled", "replacement", "redact_tool_input",
            "redact_tool_response", "redact_reasoning",
            "patterns", "sensitive_keys",
        ):
            assert key in d

    def test_patterns_match_builtins(self):
        d = build_default_redaction_dict()
        builtin_names = {p["name"] for p in BUILTIN_PATTERNS}
        dict_names = {p["name"] for p in d["patterns"]}
        assert dict_names == builtin_names

    def test_no_internal_keys_leaked(self):
        d = build_default_redaction_dict()
        for p in d["patterns"]:
            for key in p:
                assert not key.startswith("_"), f"internal key {key} leaked"

    def test_sensitive_keys_sorted(self):
        d = build_default_redaction_dict()
        assert d["sensitive_keys"] == sorted(d["sensitive_keys"])

    def test_roundtrip_through_redaction_config(self):
        d = build_default_redaction_dict()
        cfg = RedactionConfig.from_dict(d)
        assert cfg.enabled is True
        # Should detect secrets
        text = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        assert "[REDACTED]" in redact_string(text, cfg)
        # Should scrub keys
        data = {"password": "secret"}
        assert redact_dict(data, cfg)["password"] == "[REDACTED]"
