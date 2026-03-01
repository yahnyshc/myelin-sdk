"""Following a matched workflow with step-by-step hints.

When Myelin finds a matching workflow (HIT), the agent receives
an overview and skeleton. It can request detailed hints for each
step, adapting the proven procedure to the current context.

Usage:
    export MYELIN_API_KEY=my_...
    export OPENAI_API_KEY=sk-...
    python workflow_follower.py
"""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from myelin_sdk import MyelinSession


@tool
def search_users(query: str) -> str:
    """Search for users by name, email, or ID."""
    return '[{"id": "usr_42", "name": "Alice Chen", "email": "alice@example.com", "plan": "pro"}]'


@tool
def get_billing_info(user_id: str) -> str:
    """Get billing details for a user."""
    return '{"plan": "pro", "mrr": 49.00, "next_invoice": "2026-03-01", "payment_method": "visa-4242"}'


@tool
def apply_credit(user_id: str, amount: float, reason: str) -> str:
    """Apply account credit to a user."""
    return f'{{"status": "applied", "amount": {amount}, "new_balance": {amount}}}'


@tool
def send_reply(ticket_id: str, message: str) -> str:
    """Send a reply to a support ticket."""
    return f'{{"status": "sent", "ticket_id": "{ticket_id}"}}'


TOOLS = [search_users, get_billing_info, apply_credit, send_reply]
TOOL_MAP = {t.name: t for t in TOOLS}


async def run_agent(llm, messages: list, handler) -> str:
    """Simple agent loop: call LLM, execute tools, repeat."""
    agent = llm.bind_tools(TOOLS)

    while True:
        response = await agent.ainvoke(messages, config={"callbacks": [handler]})
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tc in response.tool_calls:
            result = await TOOL_MAP[tc["name"]].ainvoke(tc["args"])
            from langchain_core.messages import ToolMessage

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))


async def main():
    ticket_msg = (
        "Ticket #5678: alice@example.com was charged twice for January. "
        "Please apply a $49 credit and let them know."
    )

    async with MyelinSession.start("handle a billing credit request") as session:
        handler = session.callback()
        llm = ChatOpenAI(model="gpt-4o-mini")

        if session.matched:
            wf = session.workflow
            print(f"Matched workflow: {wf.description}")
            print(f"Success-proven procedure with {wf.total_steps} steps\n")

            system_prompt = await session.build_system_prompt()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=ticket_msg),
            ]
        else:
            print("No matching workflow found — working freestyle")
            print("Myelin will record this session and extract a workflow if successful\n")

            messages = [
                SystemMessage(
                    content=(
                        "You are a billing support agent. Use the available tools "
                        "to resolve the customer's issue."
                    )
                ),
                HumanMessage(content=ticket_msg),
            ]

        response = await run_agent(llm, messages, handler)
        print(f"Agent: {response}\n")

        result = await session.finish()
        print(f"Recorded {result.tool_calls_recorded} tool calls")
        if result.workflow_id:
            print(f"New workflow extracted: {result.workflow_id}")


if __name__ == "__main__":
    asyncio.run(main())
