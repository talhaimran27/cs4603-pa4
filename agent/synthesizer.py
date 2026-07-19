"""Synthesizer node (Task 1.6).

TODO: Implement `make_synthesizer(llm)` returning a node that combines
step_results into one cited answer and writes it to BOTH `final_answer` AND
the `messages` channel as an AIMessage (required for the OpenAI-compatible
serving contract — see spec Task 1.6).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def _content_as_text(value: Any) -> str:
    if hasattr(value, "content"):
        value = value.content

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts = []

        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")

                if text:
                    parts.append(str(text))

        return "\n".join(parts)

    return str(value)


def _latest_user_question(messages: list) -> str:
    for message in reversed(messages):
        message_type = getattr(message, "type", None)
        role = message.get("role") if isinstance(message, dict) else None

        if message_type == "human" or role in {"user", "human"}:
            return _content_as_text(message).strip()

    return ""




def make_synthesizer(llm):
    def synthesizer(state: AnalystState) -> dict:
        question = _latest_user_question(
            state.get("messages", [])
        )

        plan = state.get("plan", [])
        results = state.get("step_results", [])

        combined_steps: list[str] = []

        max_length = max(len(plan), len(results))

        for index in range(max_length):
            step = (
                plan[index]
                if index < len(plan)
                else "Unspecified step"
            )
            result = (
                results[index]
                if index < len(results)
                else "No result produced"
            )

            combined_steps.append(
                f"Step {index + 1}\n"
                f"Task: {step}\n"
                f"Result: {result}"
            )

        execution_context = "\n\n".join(combined_steps)

        response = llm.invoke(
            [
                SystemMessage(content=SYNTHESIZER_PROMPT),
                HumanMessage(
                    content=(
                        f"Original user question:\n{question}\n\n"
                        f"Completed step results:\n"
                        f"{execution_context}"
                    )
                ),
            ]
        )

        answer = _content_as_text(response).strip()

        if not answer:
            answer = (
                "I could not produce a final answer from the "
                "available step results."
            )

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    return synthesizer
