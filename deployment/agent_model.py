"""MLflow models-from-code definition (Task 2.1).

TODO: Make this file self-contained so MLflow can serialise it:
  - validate DATABRICKS_HOST/TOKEN/MODEL at import time (clear error if missing),
  - rebuild the graph with production clients (LLM, Vector Search retriever,
    MCP tools),
  - end with `mlflow.models.set_model(graph)`.

Must import cleanly:  python -c "import deployment.agent_model"
"""

from __future__ import annotations

# TODO: import os, mlflow, build_graph, get_chat_llm, get_retriever, load_mcp_tools
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from agent.graph import build_graph, load_mcp_tools
from config import get_chat_llm
from rag.store import get_retriever
# TODO: validate env vars
load_dotenv()


REQUIRED_ENV_VARS = [
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_MODEL",
    "VECTOR_SEARCH_ENDPOINT",
    "VECTOR_SEARCH_INDEX",
    "EMBEDDINGS_ENDPOINT",
]


def validate_environment() -> None:
    """Raise a clear startup error when required configuration is missing."""
    missing = [
        variable
        for variable in REQUIRED_ENV_VARS
        if not os.getenv(variable, "").strip()
    ]

    if missing:
        missing_text = ", ".join(missing)

        raise RuntimeError(
            "Document Analyst model could not start because the following "
            f"required environment variables are missing: {missing_text}. "
            "Configure them locally in .env or provide them through the "
            "Databricks Model Serving endpoint environment variables."
        )


validate_environment()

# TODO: graph = build_graph(...)

# TODO: mlflow.models.set_model(graph)


# Create production dependencies once when the model is loaded.
llm = get_chat_llm(temperature=0.0)
retriever = get_retriever(k=4)
tools = load_mcp_tools()


# Rebuild the complete LangGraph application.
graph = build_graph(
    llm=llm,
    retriever=retriever,
    tools=tools,
)


# Expose the compiled graph to MLflow models-from-code.
mlflow.models.set_model(graph)
