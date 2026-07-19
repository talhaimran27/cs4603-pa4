"""Supervisor node + routing edge (Task 1.3).

TODO:
  - `make_supervisor(llm)`: if current_step_index >= len(plan) -> next_agent =
    'synthesizer'; else classify the current step to 'rag_agent' or 'mcp_tools'.
  - `route_from_supervisor(state)`: return state["next_agent"] for the
    conditional edge.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"



def _content_as_text(value: Any) -> str:
    if hasattr(value, "content"):
        value = value.content

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts = []

        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")

                if text:
                    parts.append(str(text))

        return "\n".join(parts)

    return str(value)


def _fallback_route(step: str) -> str:
    """Keyword fallback when the LLM returns an invalid label."""
    calculation_terms = {
        "calculate",
        "compute",
        "percentage",
        "percent",
        "growth",
        "cagr",
        "increase",
        "decrease",
        "difference",
        "compare",
        "convert",
        "ratio",
        "multiply",
        "divide",
        "sum",
        "average",
        "after",
        "projection",
        "projected",
    }

    normalized = step.lower()

    if any(term in normalized for term in calculation_terms):
        return MCP

    return RAG

def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        index = state.get("current_step_index", 0)

        #Deterministic rule
        if index >=len(plan):
            return {"next_agent" : SYNTH}
        
        current_step = plan[index]

        response = llm.invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=f"Current step:\n{current_step}")
            ]
        )

        decision = _content_as_text(response).strip().lower()

        # Accept output such as "rag_agent." or explanatory text.
        if RAG in decision:
            route = RAG
        elif MCP in decision:
            route = MCP
        else:
            route = _fallback_route(current_step)

        return {"next_agent": route}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    route = state.get("next_agent", "")

    allowed = {RAG, MCP, SYNTH}

    if route not in allowed:
        raise ValueError(
            f"Invalid supervisor route {route!r}. "
            f"Expected one of {sorted(allowed)}."
        )
    return route
