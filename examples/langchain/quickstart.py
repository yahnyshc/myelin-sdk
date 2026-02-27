"""Quickstart: Add Myelin memory to a LangChain agent in 5 lines.

The agent handles a support ticket. Myelin records the tool calls
so it can extract a reusable workflow from this session.

Usage:
    export MYELIN_API_KEY=my_...
    export OPENAI_API_KEY=sk-...
    python quickstart.py
"""

import asyncio
import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from myelin_sdk import MyelinClient
from myelin_sdk.langchain import MyelinCallbackHandler


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


async def main():
    # 1. Initialize Myelin
    myelin = MyelinClient(api_key=os.environ["MYELIN_API_KEY"])
    recall = await myelin.recall("handle a password reset support ticket")
    print(f"Recall: session={recall.session_id}, matched={recall.matched}")
    if recall.matched:
        print(f"  Workflow: {recall.workflow.description}")
        print(f"  Steps: {recall.workflow.total_steps}")
        print(f"  Overview: {recall.workflow.overview[:200]}...")
        print(f"  Skeleton:\n{recall.workflow.skeleton}")

    # 2. Create callback handler — this is the only integration point
    handler = MyelinCallbackHandler(client=myelin, session_id=recall.session_id)

    # 3. Build your agent as usual, just pass the handler
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = llm.bind_tools([lookup_user, reset_password, send_reply])

    # 4. Run the agent with the callback
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = []

    # If Myelin found a matching workflow, give the agent its guidance
    if recall.matched:
        messages.append(SystemMessage(content=recall.workflow.overview))

    messages.append(
        HumanMessage(
            content=(
                "Ticket #1234: Customer alice@example.com says they can't log in. "
                "Please reset their password and let them know."
            )
        )
    )

    # Agent loop — invoke until no more tool calls
    while True:
        response = await agent.ainvoke(messages, config={"callbacks": [handler]})
        messages.append(response)

        if not response.tool_calls:
            break

        # Execute tool calls
        for tc in response.tool_calls:
            tool_fn = {"lookup_user": lookup_user, "reset_password": reset_password, "send_reply": send_reply}[tc["name"]]
            result = await tool_fn.ainvoke(tc["args"], config={"callbacks": [handler]})
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    print(f"Agent response: {response.content}")

    # 5. Debrief — Myelin evaluates the session and extracts a workflow
    result = await myelin.debrief(recall.session_id)
    print(f"Session {result.session_id}: {result.tool_calls_recorded} tool calls recorded")

    await myelin.close()


if __name__ == "__main__":
    asyncio.run(main())
