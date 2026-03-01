"""ReAct research agent that learns from experience.

A research agent with web search and note-taking tools. Over multiple
runs, Myelin extracts workflows from successful research sessions
so the agent gets better at structured research over time.

Uses LangGraph's prebuilt ReAct agent for a production-ready setup.

Usage:
    export MYELIN_API_KEY=my_...
    export OPENAI_API_KEY=sk-...
    pip install langgraph langchain-openai
    python react_research_agent.py
"""

import asyncio

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from myelin_sdk import MyelinSession


# --- Tools ---

_notes: list[str] = []


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Replace with a real search tool (e.g., Tavily, SerpAPI)
    return (
        f'Results for "{query}":\n'
        "1. Myelin is a procedural memory system for AI agents\n"
        "2. It uses MCP protocol for tool integration\n"
        "3. Workflows are extracted from successful agent sessions\n"
    )


@tool
def read_url(url: str) -> str:
    """Read the contents of a URL."""
    return f"Content from {url}: This is a mock response. Replace with real HTTP fetch."


@tool
def take_note(content: str) -> str:
    """Save a research note for the final report."""
    _notes.append(content)
    return f"Note saved ({len(_notes)} total)"


@tool
def write_report(title: str, sections: str) -> str:
    """Write the final research report. Sections should be newline-separated."""
    notes_text = "\n".join(f"- {n}" for n in _notes) if _notes else "(no notes)"
    return (
        f"# {title}\n\n"
        f"## Research Notes\n{notes_text}\n\n"
        f"## Report\n{sections}"
    )


async def main():
    task = "research how procedural memory differs from semantic memory in AI agents"

    async with MyelinSession.start(task) as session:
        if session.matched:
            print(f"Following workflow: {session.workflow.description}")
            print(f"Steps: {session.workflow.skeleton}\n")
        else:
            print("No existing workflow — pioneering a new research approach\n")

        handler = session.callback()
        llm = ChatOpenAI(model="gpt-4o-mini")

        agent = create_react_agent(
            llm,
            tools=[web_search, read_url, take_note, write_report],
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": task}]},
            config={"callbacks": [handler]},
        )

        final = result["messages"][-1]
        print(f"Agent: {final.content}\n")

        result = await session.finish()
        print(f"Recorded {result.tool_calls_recorded} tool calls")
        if result.workflow_id:
            print(f"Workflow extracted: {result.workflow_id}")
        print(
            "\nNext time this task is run, Myelin may return a proven workflow "
            "so the agent follows a structured research approach."
        )


if __name__ == "__main__":
    asyncio.run(main())
