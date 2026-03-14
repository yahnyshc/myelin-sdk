# Myelin SDK

Procedural memory for AI agents. Agents that have done a task 100 times shouldn't fumble on attempt 101.

## Claude Code

Zero-code integration via PostToolUse hooks.

### 1. Install the SDK

```bash
uv tool install myelin-sdk
```

### 2. Add the MCP server

```bash
claude mcp add --scope project --transport http myelin https://myelin.fly.dev/mcp \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 3. Add the PostToolUse hook

Add this to `.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "myelin-capture"
          }
        ]
      }
    ],
    "PostToolUseFailure": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "myelin-capture"
          }
        ]
      }
    ]
  }
}
```

### 4. Update `.gitignore`

Add `.mcp.json` to your `.gitignore` (it contains your API key).

The hook captures every tool call automatically. Use `memory.search` to find workflows, `memory.start` to begin a session, and `memory.finish` to end it.

## Python SDK / LangChain

Explicit integration for LangChain and LangGraph agents.

```bash
pip install myelin-sdk[langchain]
```

```python
from myelin_sdk import MyelinSession

async with MyelinSession.create("handle support ticket", api_key="my_...") as session:
    handler = session.langchain_handler()

    # Pass handler to your LangChain agent
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "..."}]},
        config={"callbacks": [handler]},
    )
# session.finish() called automatically on exit
```

## Adding an Integration

The `integrations/langchain/` directory is the template for new integrations. To add support for another framework (e.g., CrewAI, AutoGen):

1. Create `src/myelin_sdk/integrations/<framework>/`
2. Add `__init__.py` and a handler module
3. Use `MyelinClient.capture()` to record tool calls
4. Add an optional dependency group in `pyproject.toml`
5. Add a convenience method on `MyelinSession`

See `integrations/langchain/handler.py` for a complete reference implementation.

## Syncing Local Procedures

Sync markdown procedure files from your repo to Myelin:

```bash
pip install myelin-sdk

# Sync all procedures from default directory (.claude/procedures/*.md)
myelin-sync

# Sync from a custom directory
myelin-sync --dir ./runbooks

# Sync specific files
myelin-sync deploy.md hotfix.md
```

Requires `MYELIN_API_KEY`, `MYELIN_BASE_URL`, and `MYELIN_PROJECT_ID` environment variables (or reads from `.mcp.json`).

Sync is idempotent — running twice with unchanged files produces no updates. Deleting a local file does not remove the server copy.
