"""Full Document Analyst graph (Tasks 1.5 + 1.7).

TODO:
  - `load_mcp_tools(server_path=None)`: connect the GIVEN MCP server over stdio
    (see langchain-mcp-adapters) and return its tools.
  - `make_mcp_node(tools, llm)`: execute one calculation step by letting the LLM
    call exactly one MCP tool, then append the result and increment the index.
  - `build_graph(llm=None, retriever=None, tools=None)`: assemble
    planner -> supervisor -> {rag_agent | mcp_tools} -> ... -> synthesizer.
    Inject dependencies so the graph can be unit-tested offline with fakes.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
from collections.abc import Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer


def _content_as_text(value: Any) -> str:
    """Convert an MCP/LangChain response into plain text."""

    if hasattr(value, "content"):
        value = value.content

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts: list[str] = []

        for block in value:
            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict):
                text = (
                    block.get("text")
                    or block.get("content")
                )

                if text:
                    parts.append(str(text))

            elif hasattr(block, "text"):
                parts.append(str(block.text))

        return "\n".join(parts)

    return str(value)



def _run_coroutine_sync(coroutine: Coroutine):
    """Run an async coroutine from scripts, notebooks, or serving."""

    def run_in_new_loop():
        return asyncio.run(coroutine)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run_in_new_loop()

    result_container: dict[str, Any] = {}
    error_container: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result_container["result"] = (
                run_in_new_loop()
            )
        except BaseException as exc:
            error_container["error"] = exc

    thread = threading.Thread(
        target=runner,
        daemon=True,
    )
    thread.start()
    thread.join()

    if "error" in error_container:
        raise error_container["error"]

    return result_container["result"]

def _patch_mcp_stdio_stderr() -> Path:
    """Make MCP stdio subprocesses use a real stderr file in serving."""

    import langchain_mcp_adapters.sessions as mcp_sessions
    from mcp.client.stdio import stdio_client as original_stdio_client

    log_path = (
        Path(tempfile.gettempdir())
        / "pa4_mcp_server.log"
    )

    @asynccontextmanager
    async def safe_stdio_client(server):
        # Open a real file for every MCP session. This is necessary because
        # Databricks Model Serving replaces sys.stderr with StreamToLogger,
        # which does not expose fileno().
        with open(
            log_path,
            "a",
            encoding="utf-8",
        ) as error_log:
            async with original_stdio_client(
                server,
                errlog=error_log,
            ) as streams:
                yield streams

    # langchain-mcp-adapters imported stdio_client into this module, so patch
    # the exact reference that its session factory calls.
    mcp_sessions.stdio_client = safe_stdio_client

    return log_path

def load_mcp_tools(server_path: str | None = None):
    """Launch the supplied MCP server over stdio and load its tools."""

    if server_path is None:
        project_root = Path(__file__).resolve().parents[1]

        server_path = str(
            project_root
            / "tools"
            / "mcp_server.py"
        )

    resolved_path = Path(server_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"MCP server not found at {resolved_path}"
        )

    # Patch the stdio transport before MultiServerMCPClient opens a session.
    log_path = _patch_mcp_stdio_stderr()

    server_config = {
        "command": sys.executable,
        "args": [
            "-u",
            str(resolved_path),
        ],
        "transport": "stdio",
        "env": {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        },
    }

    client = MultiServerMCPClient(
        {
            "analyst_math": server_config,
        }
    )

    tools = _run_coroutine_sync(
        client.get_tools()
    )

    if not tools:
        raise RuntimeError(
            "The MCP server returned no tools. "
            f"Check the MCP log at {log_path}"
        )

    print(f"Loaded {len(tools)} MCP tool(s):")

    for tool in tools:
        print(f"  - {tool.name}")

    print(f"MCP subprocess log: {log_path}")

    return tools


def _tool_map(tools) -> dict[str, Any]:
    return {tool.name: tool for tool in tools}



def make_mcp_node(tools, llm):
    """Create the MCP calculation specialist node."""
    tools_by_name = _tool_map(tools)
    llm_with_tools = llm.bind_tools(tools)

    def mcp_tools(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        index = state.get("current_step_index", 0)

        if index >= len(plan):
            raise IndexError(
                "MCP node was called after all plan steps were complete."
            )

        current_step = plan[index]
        previous_results = state.get("step_results", [])

        prior_context = "\n".join(
            f"Step {position}: {result}"
            for position, result in enumerate(
                previous_results,
                start=1,
            )
        )

        if not prior_context:
            prior_context = "No previous step results."

        response = llm_with_tools.invoke(
            [
                SystemMessage(content=MCP_STEP_PROMPT),
                HumanMessage(
                    content=(
                        f"Current calculation step:\n{current_step}\n\n"
                        f"Previous step results:\n{prior_context}"
                    )
                ),
            ]
        )

        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            result = (
                "calculation failed: the model did not call an MCP tool"
            )
        else:
            # Enforce the assignment requirement: exactly one tool call.
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            selected_tool = tools_by_name.get(tool_name)

            if selected_tool is None:
                result = (
                    f"calculation failed: unknown MCP tool "
                    f"{tool_name!r}"
                )
            else:
                try:
                    raw_result =  _run_coroutine_sync(selected_tool.ainvoke(tool_args))
                except Exception as exc:
                    result = (
                        "calculation failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    result = _content_as_text(raw_result).strip()

                if not result:
                    result = "calculation failed: empty tool response"

        return {
            "step_results": [
                *previous_results,
                result,
            ],
            "current_step_index": index + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    if llm is None:
        from config import get_chat_llm

        llm = get_chat_llm(temperature=0.0)

    if retriever is None:
        from rag.store import get_retriever

        retriever = get_retriever(k=4)

    if tools is None:
        tools = load_mcp_tools()

    planner = make_planner(llm)
    supervisor = make_supervisor(llm)
    rag_agent = make_rag_agent(retriever, llm)
    mcp_tools = make_mcp_node(tools, llm)
    synthesizer = make_synthesizer(llm)

    builder = StateGraph(AnalystState)

    builder.add_node("planner", planner)
    builder.add_node("supervisor", supervisor)
    builder.add_node("rag_agent", rag_agent)
    builder.add_node("mcp_tools", mcp_tools)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            RAG: "rag_agent",
            MCP: "mcp_tools",
            SYNTH: "synthesizer",
        },
    )

    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("mcp_tools", "supervisor")
    builder.add_edge("synthesizer", END)

    return builder.compile()
