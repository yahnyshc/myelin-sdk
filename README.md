# Myelin SDK

Procedural memory for AI agents. Capture what works, surface it on the next task, and improve it automatically.

Myelin records your agent's tool calls, evaluates outcomes, and extracts reusable workflows from successful sessions. Next time a similar task comes up, your agent gets a proven procedure instead of starting from scratch.

```
search → find matching workflow
record → begin session, capture tool calls
finish → finalize, queue for evaluation
```

## Install

```bash
pip install myelin-sdk
```

Get an API key from [myelin.vercel.app](https://myelin.vercel.app) — free tier includes 50 sessions/month.

## Claude Code (zero-code capture)

The fastest path. No code changes — just add the MCP server and a PostToolUse hook.

### 1. Install the SDK

```bash
uv tool install myelin-sdk
```

### 2. Add the MCP server

```bash
claude mcp add --scope project --transport http myelin https://myelin.fly.dev/mcp \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 3. Add the capture hook

Add to `.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "",
        "hooks": [{ "type": "command", "command": "myelin-capture" }]
      }
    ],
    "PostToolUseFailure": [
      {
        "matcher": "",
        "hooks": [{ "type": "command", "command": "myelin-capture" }]
      }
    ]
  }
}
```

> If `myelin-capture` is not on your PATH, use `python -m myelin_sdk.claude_code` instead.

### 4. Add `.mcp.json` to `.gitignore`

It contains your API key.

That's it. The hook captures every tool call automatically. Use the MCP tools to control sessions:

- **`search`** — find matching workflows (read-only, no session created)
- **`record`** — begin a recording session (with a workflow ID or freestyle)
- **`finish`** — finalize the session and queue for evaluation

## Python SDK (LangChain / LangGraph)

For explicit integration with LangChain, LangGraph, or any Python agent.

```bash
pip install myelin-sdk[langchain]
```

### Quickstart

```python
import asyncio
from myelin_sdk import MyelinSession

async def main():
    async with MyelinSession.create(
        "handle a password reset ticket",
        api_key="mk_...",
    ) as session:
        handler = session.langchain_handler()

        # Pass handler as a callback to your LangChain agent
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "..."}]},
            config={"callbacks": [handler]},
        )
    # session.finish() called automatically on context manager exit

asyncio.run(main())
```

### With MyelinToolkit (agent-controlled)

Give the agent search/record/finish as tools — it decides when to use them.

```python
from myelin_sdk.integrations.langchain import MyelinToolkit

async with MyelinToolkit(api_key="mk_...") as tk:
    all_tools = tk.tools + your_tools

    agent = llm.bind_tools(all_tools)
    response = await agent.ainvoke(
        messages,
        config={"callbacks": [tk.handler]},
    )
```

### Direct capture (no framework)

Use `MyelinSession` directly if you're not using LangChain:

```python
async with MyelinSession.create("deploy to production", api_key="mk_...") as session:
    # Manually capture tool calls
    await session.capture(
        tool_name="run_deploy",
        tool_input={"env": "production"},
        tool_response="Deployed successfully to prod-east-1",
    )

    # Add notes during the session
    await session.feedback("Chose blue-green deployment due to database migration")

    result = await session.finish()
    print(f"Recorded {result.tool_calls_recorded} tool calls")
```

## Syncing local procedures

Push markdown procedure files from your repo to Myelin:

```bash
# Sync all procedures from default directory (.claude/procedures/*.md)
myelin-sync

# Sync from a custom directory
myelin-sync --dir ./runbooks

# Sync specific files
myelin-sync deploy.md hotfix.md
```

Credentials are read from `MYELIN_API_KEY` and `MYELIN_BASE_URL` environment variables, or from `.mcp.json`.

Sync is idempotent — unchanged files produce no updates.

## Redaction

Myelin automatically redacts sensitive data before sending anything to the server. Built-in patterns cover API keys, tokens, passwords, database URLs, and 20+ other secret formats.

To customize:

```python
from myelin_sdk import RedactionConfig

# Add custom patterns
config = RedactionConfig(
    additional_patterns=[{"name": "internal_token", "pattern": r"itk_[a-zA-Z0-9]{32}"}],
    additional_keys=["x-internal-secret"],
)

session = MyelinSession.create("...", api_key="...", redaction=config)
```

Or configure via file (auto-discovered at `.claude/hooks/redaction.json`):

```json
{
  "additional_patterns": [
    {"name": "internal_token", "pattern": "itk_[a-zA-Z0-9]{32}"}
  ],
  "additional_keys": ["x-internal-secret"]
}
```

Disable redaction entirely with `MYELIN_REDACT=0` (not recommended).

## How it works

1. **Agent searches** — Myelin finds the best matching workflow via semantic search
2. **Agent records** — Session starts, tool calls are captured automatically
3. **Agent finishes** — Session is queued for background evaluation
4. **Evaluation** — LLM judge assigns a verdict: `success`, `partial`, or `failure`
5. **Extraction** — Successful sessions get workflows extracted automatically
6. **Approval** — You review and approve extracted workflows in the [dashboard](https://myelin.vercel.app)
7. **Next run** — The approved workflow is returned on the next matching search

The flywheel: agents handle tasks → traces are stored → workflows are extracted → success rates update → next agent gets proven procedures → performs better → repeat.

## API reference

### MyelinSession

| Method | Description |
|---|---|
| `MyelinSession.create(task, *, api_key, base_url, workflow_id, project_id, redaction)` | Create a session (async context manager) |
| `session.session_id` | The active session ID |
| `session.matched_workflow_id` | Workflow ID if a match was found, else `None` |
| `session.capture(tool_name, tool_input, tool_response, context, client_ts)` | Record a tool call |
| `session.feedback(notes)` | Append a timestamped note to the session |
| `session.finish()` | Finalize and queue for evaluation |
| `session.langchain_handler(*, hide_inputs, hide_outputs, redaction)` | Get a LangChain callback handler |

### Response types

```python
from myelin_sdk import (
    SearchMatch,      # workflow_id, title, description, content, score
    SearchResult,     # top_match, other_matches
    StartResult,      # session_id, matched_workflow_id
    CaptureResponse,  # status
    FeedbackResponse, # session_id, status
    FinishResponse,   # session_id, tool_calls_recorded, status, workflow_id
    SyncResult,       # total, created, updated, unchanged, details
)
```

## Links

- [Dashboard](https://myelin.vercel.app)
- [GitHub](https://github.com/yahnyshc/myelin-sdk)
- [Issues](https://github.com/yahnyshc/myelin-sdk/issues)

## License

MIT
