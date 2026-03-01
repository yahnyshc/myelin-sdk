"""Pytest configuration for myelin-sdk tests."""

import sys
import types

import pytest


class _AsyncCallbackHandler:
    pass


class _BaseTool:
    """Minimal BaseTool mock that supports Pydantic-style class vars."""

    def __init__(self, **kwargs):
        # Accept and ignore keyword args that real BaseTool would handle
        pass


@pytest.fixture
def patch_langchain(monkeypatch):
    """Patch langchain_core imports so tests work without langchain installed."""
    langchain_core = types.ModuleType("langchain_core")
    callbacks_mod = types.ModuleType("langchain_core.callbacks")
    outputs_mod = types.ModuleType("langchain_core.outputs")
    tools_mod = types.ModuleType("langchain_core.tools")

    callbacks_mod.AsyncCallbackHandler = _AsyncCallbackHandler
    outputs_mod.LLMResult = type("LLMResult", (), {})
    tools_mod.BaseTool = _BaseTool

    langchain_core.callbacks = callbacks_mod
    langchain_core.outputs = outputs_mod
    langchain_core.tools = tools_mod

    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", callbacks_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.outputs", outputs_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", tools_mod)

    # Force reimport of handler module so it picks up our mocks
    for mod_name in list(sys.modules):
        if "myelin_sdk.integrations.langchain" in mod_name:
            monkeypatch.delitem(sys.modules, mod_name)
