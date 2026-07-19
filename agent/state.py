"""State schema for the Document Analyst graph (Task 1.1).

TODO: Define `AnalystState` as a TypedDict with the fields from the spec table:
  messages, plan, current_step_index, step_results, next_agent, final_answer.
Use `Annotated[list, add_messages]` for `messages`.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AnalystState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    # TODO: plan, current_step_index, step_results, next_agent, final_answer
    plan: list[str]
    current_step_index: int
    step_results: list[str]
    next_agent: str
    final_answer: str
