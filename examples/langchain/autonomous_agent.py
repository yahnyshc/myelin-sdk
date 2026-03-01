"""Autonomous agent: Myelin recall/hint/finish as LangChain tools.

The agent decides when to search for workflows, request step details,
and finalize the session — no developer orchestration needed.

Usage:
    export MYELIN_API_KEY=my_...
    export OPENAI_API_KEY=sk-...
    python autonomous_agent.py
"""

import asyncio
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from myelin_sdk.integrations.langchain import MyelinToolkit


# --- Mock tools (replace with your real tools) ---


@tool
def lookup_user(email: str) -> str:
    """Look up a user by email address."""
    return f'{{"id": "usr_42", "name": "Alice", "email": "{email}", "plan": "pro"}}'


@tool
def reset_password(user_id: str) -> str:
    """Reset a user's password and send a reset link."""
    return f'{{"status": "ok", "user_id": "{user_id}", "reset_link_sent": true}}'


@tool
def send_reply(ticket_id: str, message: str) -> str:
    """Send a reply to a support ticket."""
    return f'{{"status": "sent", "ticket_id": "{ticket_id}"}}'


# --- Main ---


SYSTEM_PROMPT = """\
You are a support agent. Before starting any task, call memory_recall with a \
description of what you're about to do. If a workflow is found, follow it and \
use memory_hint for step details. When done, call memory_finish.
"""


async def main():
    api_key = os.environ.get("MYELIN_API_KEY", "")
    if not api_key:
        print("Set MYELIN_API_KEY to run this example.")
        return

    async with MyelinToolkit(api_key=api_key) as tk:
        # Combine Myelin tools with your domain tools
        all_tools = tk.tools + [lookup_user, reset_password, send_reply]

        llm = ChatOpenAI(model="gpt-4o-mini")
        agent = llm.bind_tools(all_tools)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Ticket #1234: Customer alice@example.com says they can't "
                    "log in. Please reset their password and let them know."
                )
            ),
        ]

        # Agent loop
        from langchain_core.messages import ToolMessage

        tool_map = {t.name: t for t in all_tools}

        while True:
            response = await agent.ainvoke(
                messages, config={"callbacks": [tk.handler]}
            )
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                tool_fn = tool_map[tc["name"]]
                result = await tool_fn.ainvoke(
                    tc["args"], config={"callbacks": [tk.handler]}
                )
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

        print(f"Agent response: {response.content}")
    # MyelinToolkit auto-finishes if the agent forgot to call memory_finish


if __name__ == "__main__":
    asyncio.run(main())
