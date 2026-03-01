"""Quickstart: Add Myelin memory to a LangChain agent in 5 lines.

The agent handles a support ticket. Myelin records the tool calls
so it can extract a reusable workflow from this session.

Usage:
    export MYELIN_API_KEY=my_...
    export OPENAI_API_KEY=sk-...
    python quickstart.py
"""

import asyncio

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from myelin_sdk import MyelinSession


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
    async with MyelinSession.start("handle a password reset support ticket") as session:
        print(f"Recall: session={session.session_id}, matched={session.matched}")
        if session.matched:
            print(f"  Workflow: {session.workflow.description}")
            print(f"  Steps: {session.workflow.total_steps}")

        # Create callback handler — this is the only integration point
        handler = session.langchain_handler()

        # Build your agent as usual, just pass the handler
        llm = ChatOpenAI(model="gpt-4o-mini")
        agent = llm.bind_tools([lookup_user, reset_password, send_reply])

        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        messages = []

        # If Myelin found a matching workflow, give the agent its guidance
        if session.matched:
            messages.append(SystemMessage(content=session.workflow.overview))

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

            for tc in response.tool_calls:
                tool_fn = {"lookup_user": lookup_user, "reset_password": reset_password, "send_reply": send_reply}[tc["name"]]
                result = await tool_fn.ainvoke(tc["args"], config={"callbacks": [handler]})
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        print(f"Agent response: {response.content}")
    # Finish happens automatically when the context manager exits


if __name__ == "__main__":
    asyncio.run(main())
