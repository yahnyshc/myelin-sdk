"""Pytest configuration for myelin-sdk tests."""

import sys
import types

import pytest


class _AsyncCallbackHandler:
    pass


@pytest.fixture
def patch_langchain(monkeypatch):
    """Patch langchain_core imports so tests work without langchain installed."""
    langchain_core = types.ModuleType("langchain_core")
    callbacks_mod = types.ModuleType("langchain_core.callbacks")
    outputs_mod = types.ModuleType("langchain_core.outputs")

    callbacks_mod.AsyncCallbackHandler = _AsyncCallbackHandler
    outputs_mod.LLMResult = type("LLMResult", (), {})

    langchain_core.callbacks = callbacks_mod
    langchain_core.outputs = outputs_mod

    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", callbacks_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.outputs", outputs_mod)

    # Force reimport of handler module so it picks up our mocks
    for mod_name in list(sys.modules):
        if "myelin_sdk.integrations.langchain" in mod_name:
            monkeypatch.delitem(sys.modules, mod_name)
