"""Offline smoke test for the Document Analyst graph (Bonus A test target).

This is the target the Bonus A CI pipeline runs to prove the graph wires up
before any deploy. Fill it in once your nodes are implemented.

TODO (Task 1.7 / Bonus A):
  - Build fake LLM / retriever / tool objects (no Databricks, no network).
  - Call `build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])`.
  - Invoke it on a combined retrieval+calculation query and assert that a plan was
    produced, both specialists ran, and the final answer surfaced on messages[-1].

Run:  uv run pytest -q
"""

from __future__ import annotations

import os
import sys

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeRetriever:
    """Returns a fixed financial-document chunk."""

    def __init__(self) -> None:
        self.called = False
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.called = True
        self.queries.append(query)

        return [
            Document(
                page_content=(
                    "Meridian Motor Corporation reported net revenue "
                    "of 16.91 trillion yen in fiscal year 2023."
                ),
                metadata={
                    "source": "annual_report.pdf",
                    "page": 4,
                },
            )
        ]


class FakeTool:
    """Fake replacement for the MCP growth_rate tool."""

    name = "growth_rate"
    description = "Calculate compound annual growth."

    def __init__(self) -> None:
        self.called = False
        self.arguments: dict[str, Any] | None = None

    async def ainvoke(self, arguments: dict[str, Any]) -> str:
        self.called = True
        self.arguments = arguments

        start_value = float(arguments["start_value"])
        rate = float(arguments["rate"])
        years = int(arguments["years"])

        result = start_value * ((1 + rate) ** years)

        return (
            f"{start_value:.2f} trillion yen grown at "
            f"{rate * 100:.0f}% for {years} years equals "
            f"{result:.2f} trillion yen."
        )


class FakeLLM:
    """Returns deterministic responses for each graph node."""

    def __init__(self) -> None:
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "FakeLLM":
        self.bound_tools = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        system_prompt = str(messages[0].content)
        user_prompt = str(messages[-1].content)

        # Planner
        if "planning component" in system_prompt:
            return AIMessage(
                content=(
                    "["
                    '"Find Meridian Motor Corporation fiscal year '
                    '2023 net revenue", '
                    '"Calculate that revenue after 3 years at '
                    '8 percent compound annual growth"'
                    "]"
                )
            )

        # Supervisor
        if "routing supervisor" in system_prompt:
            if "Calculate" in user_prompt:
                return AIMessage(content="mcp_tools")

            return AIMessage(content="rag_agent")

        # RAG extraction
        if "extract one factual answer" in system_prompt:
            return AIMessage(
                content=(
                    "Meridian Motor Corporation's FY2023 net revenue "
                    "was 16.91 trillion yen "
                    "[source: annual_report.pdf, p.4]."
                )
            )

        # MCP tool selection
        if "calculation specialist" in system_prompt:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "growth_rate",
                        "args": {
                            "start_value": 16.91,
                            "rate": 0.08,
                            "years": 3,
                        },
                        "id": "fake-tool-call-1",
                        "type": "tool_call",
                    }
                ],
            )

        # Final synthesis
        if "final answer writer" in system_prompt:
            return AIMessage(
                content=(
                    "Meridian Motor Corporation's FY2023 net revenue "
                    "was 16.91 trillion yen "
                    "[source: annual_report.pdf, p.4]. "
                    "After 3 years of 8% compound annual growth, "
                    "it would be approximately 21.30 trillion yen."
                )
            )

        raise AssertionError(
            f"FakeLLM received an unexpected prompt:\n{system_prompt}"
        )


def test_graph_module_imports() -> None:
    """The graph module must import successfully."""

    from agent.graph import build_graph

    assert callable(build_graph)


def test_graph_runs_offline() -> None:
    """The full graph should run with fake offline dependencies."""

    from agent.graph import build_graph

    fake_llm = FakeLLM()
    fake_retriever = FakeRetriever()
    fake_tool = FakeTool()

    graph = build_graph(
        llm=fake_llm,
        retriever=fake_retriever,
        tools=[fake_tool],
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "What was Meridian's net revenue in FY2023, "
                        "and what would it be after 3 years of "
                        "8% compound annual growth?"
                    )
                )
            ]
        }
    )

    # The planner produced a plan.
    assert result["plan"]
    assert len(result["plan"]) == 2

    # Both plan steps completed.
    assert result["current_step_index"] == 2
    assert len(result["step_results"]) == 2

    # The RAG specialist ran.
    assert fake_retriever.called is True
    assert len(fake_retriever.queries) == 1
    assert "annual_report.pdf" in result["step_results"][0]

    # The MCP calculation specialist ran.
    assert fake_tool.called is True
    assert fake_tool.arguments is not None
    assert fake_tool.arguments["start_value"] == 16.91
    assert fake_tool.arguments["rate"] == 0.08
    assert fake_tool.arguments["years"] == 3
    assert "21.30" in result["step_results"][1]

    # The synthesizer produced the final answer.
    assert result["final_answer"].strip()

    # Most important deployment requirement:
    # the final answer must also appear in messages[-1].
    assert result["messages"]
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content.strip()
    assert result["messages"][-1].content == result["final_answer"]