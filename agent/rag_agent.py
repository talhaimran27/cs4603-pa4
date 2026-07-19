"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search.

TODO: Implement `make_rag_agent(retriever, llm)` returning a node that:
  - retrieves top-k chunks for the current step,
  - formats them with [source: file, p.N] citations,
  - extracts a single cited fact via the LLM (or 'not found in documents'),
  - appends the fact to step_results and increments current_step_index.
Reuse `rag/store.py::get_retriever()` so local and deployed retrieval match.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState


def _content_as_text(value: Any) -> str:
    if hasattr(value, "content"):
        value = value.content

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts: list[str] = []

        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")

                if text:
                    parts.append(str(text))

        return "\n".join(parts)

    return str(value)

def _citation_for_doc(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}

    source = (
        metadata.get("source")
        or metadata.get("filename")
        or metadata.get("path")
        or "unknown source"
    )

    page = (
        metadata.get("page")
        or metadata.get("page_number")
        or "unknown"
    )

    return f"[source: {source}, p.{page}]"

def format_docs(docs) -> str:
    if not docs:
        return ""

    formatted: list[str] = []

    for position, doc in enumerate(docs, start=1):
        content = getattr(doc, "page_content", str(doc)).strip()
        citation = _citation_for_doc(doc)

        formatted.append(
            f"CHUNK {position}\n"
            f"{citation}\n"
            f"{content}"
        )

    return "\n\n".join(formatted)

def make_rag_agent(retriever, llm):
    def rag_agent(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        index = state.get("current_step_index", 0)

        if index >= len(plan):
            raise IndexError(
                "RAG agent was called after all plan steps were complete."
            )

        current_step = plan[index]

        try:
            docs = retriever.invoke(current_step)
        except Exception as exc:
            # Keep the graph alive while recording the specialist failure.
            result = f"not found in documents (retrieval error: {exc})"
        else:
            if not docs:
                result = "not found in documents"
            else:
                context = format_docs(docs)

                response = llm.invoke(
                    [
                        SystemMessage(content=RAG_EXTRACT_PROMPT),
                        HumanMessage(
                            content=(
                                f"Current step:\n{current_step}\n\n"
                                f"Retrieved context:\n{context}"
                            )
                        ),
                    ]
                )

                result = _content_as_text(response).strip()

                if not result:
                    result = "not found in documents"

        updated_results = [
            *state.get("step_results", []),
            result,
        ]

        return {
            "step_results": updated_results,
            "current_step_index": index + 1,
        }

    return rag_agent
