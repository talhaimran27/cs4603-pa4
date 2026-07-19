"""Agent Framework compatible wrapper for the Document Analyst."""

from __future__ import annotations

import uuid
from typing import Any

import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

from agent.graph import build_graph, load_mcp_tools
from config import get_chat_llm
from rag.store import get_retriever


def _message_to_dict(message: Any) -> dict[str, Any]:
    """Convert an MLflow message into the graph's message format."""
    if isinstance(message, dict):
        return {
            "role": message.get("role", "user"),
            "content": message.get("content", ""),
        }

    return {
        "role": getattr(message, "role", "user"),
        "content": getattr(message, "content", ""),
    }


def _content_as_text(message: Any) -> str:
    """Extract text from LangChain or dictionary messages."""
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

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


llm = get_chat_llm(temperature=0.0)
retriever = get_retriever(k=4)
tools = load_mcp_tools()

graph = build_graph(
    llm=llm,
    retriever=retriever,
    tools=tools,
)


class DocumentAnalystChatAgent(ChatAgent):
    """ChatAgent wrapper around the existing LangGraph."""

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: ChatContext | None = None,
        custom_inputs: dict[str, Any] | None = None,
    ) -> ChatAgentResponse:
        graph_input = {
            "messages": [
                _message_to_dict(message)
                for message in messages
            ]
        }

        state = graph.invoke(graph_input)

        graph_messages = state.get("messages", [])

        if graph_messages:
            answer = _content_as_text(
                graph_messages[-1]
            ).strip()
        else:
            answer = str(
                state.get("final_answer", "")
            ).strip()

        if not answer:
            answer = (
                "The Document Analyst could not produce an answer."
            )

        return ChatAgentResponse(
            messages=[
                ChatAgentMessage(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content=answer,
                )
            ]
        )


AGENT = DocumentAnalystChatAgent()

mlflow.models.set_model(AGENT)