"""Planner node (Task 1.2).

TODO: Implement `make_planner(llm)` returning a node that:
  - reads the user question from state["messages"],
  - asks the LLM (PLANNER_PROMPT) for a JSON list of 2-5 steps,
  - parses it robustly (fallback to a single step on parse failure),
  - returns {"plan": [...], "current_step_index": 0, "step_results": []}.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState


def _message_content(message: Any) -> str:
    """Extract text from LangChain messages or dictionary messages."""
    if hasattr(message, "content"):
        content = message.content
    elif isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = str(message)

    if isinstance(content, str):
        return content

    # Some chat models return block-based content.
    if isinstance(content, list):
        parts: list[str] = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))

        return "\n".join(parts)

    return str(content)


def _latest_user_question(messages: list) -> str:
    """Return the most recent user/human message."""
    for message in reversed(messages):
        message_type = getattr(message, "type", None)
        role = message.get("role") if isinstance(message, dict) else None

        if message_type == "human" or role in {"user", "human"}:
            question = _message_content(message).strip()

            if question:
                return question

    # Defensive fallback when role metadata is unavailable.
    if messages:
        return _message_content(messages[-1]).strip()

    raise ValueError("Planner received no messages.")

def _strip_code_fence(text: str) -> str:
    """Remove a surrounding Markdown JSON fence."""
    text = text.strip()

    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return fenced.group(1).strip() if fenced else text


def _parse_plan(raw_text: str, fallback: str) -> list[str]:
    """Parse and validate an LLM-generated plan."""
    cleaned = _strip_code_fence(raw_text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to recover a JSON array embedded inside extra text.
        match = re.search(r"\[[\s\S]*\]", cleaned)

        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

    if isinstance(parsed, list):
        plan = [
            str(step).strip()
            for step in parsed
            if isinstance(step, str) and step.strip()
        ]

        if plan:
            # Enforce the assignment's maximum plan length.
            return plan[:5]

    # Graceful parse-failure fallback required by Task 1.2.
    return [fallback]


def make_planner(llm):
    """Create the LangGraph planner node."""

    def planner(state: AnalystState) -> dict:
        question = _latest_user_question(state.get("messages", []))

        response = llm.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=question),
            ]
        )

        raw_plan = _message_content(response)
        plan = _parse_plan(raw_plan, fallback=question)

        return {
            "plan": plan,
            "current_step_index": 0,
            "step_results": [],
            "next_agent": "",
            "final_answer": "",
        }

    return planner
