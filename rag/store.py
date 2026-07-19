"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

TODO: Implement `get_retriever(k=4)` that returns a LangChain retriever over the
Databricks Vector Search index built by `ingest.py`, using
`DatabricksVectorSearch` from `databricks_langchain`. Read endpoint/index names
from config.get_settings(). This exact retriever is reused by the deployed model.
"""

from __future__ import annotations
from databricks_langchain import DatabricksVectorSearch
from config import get_settings

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS = ["chunk_id", "source", "page"]

def get_vector_store():
    """Return a handle to the managed Databricks Vector Search index."""
    settings = get_settings()

    endpoint_name = settings["vs_endpoint"]
    index_name = settings["vs_index"]

    if not endpoint_name:
        raise OSError(
            "Missing VECTOR_SEARCH_ENDPOINT. "
            "Set it in your .env or serving endpoint environment variables."
        )

    if not index_name:
        raise OSError(
            "Missing VECTOR_SEARCH_INDEX. "
            "Set it in your .env or serving endpoint environment variables."
        )

    return DatabricksVectorSearch(
        endpoint=endpoint_name,
        index_name=index_name,
        columns=CITATION_COLUMNS,
    )


def get_retriever(k: int = 4):
    """Return a top-k LangChain retriever."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
