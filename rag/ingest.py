"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).

Run inside a Databricks notebook (needs Spark + ai_parse_document/ai_prep_search).
Mirror PA2 Part 1:

TODO:
  - `build_chunks_table(spark, volume_path, chunks_table)`: parse the PDF with
    ai_parse_document, chunk with ai_prep_search into a Delta table with columns
    chunk_id, chunk_to_retrieve, chunk_to_embed, source, page. Enable Change Data
    Feed on the table.
  - `create_index()`: create a STANDARD Vector Search endpoint and a TRIGGERED
    Delta Sync index (primary_key='chunk_id',
    embedding_source_column='chunk_to_retrieve',
    embedding_model_endpoint_name=$EMBEDDINGS_ENDPOINT).
"""

from __future__ import annotations

import os
import time
from typing import Any
from datetime import timedelta

from databricks.vector_search.client import VectorSearchClient


def _get_required_env(name: str) -> str:
    """Read a required environment variable.

    Raises
    ------
    EnvironmentError
        If the environment variable is missing or empty.
    """
    value = os.getenv(name)

    if not value:
        raise EnvironmentError(
            f"Required environment variable {name!r} is not configured."
        )

    return value


def _validate_table_name(table_name: str) -> None:
    """Validate that a Unity Catalog table name is fully qualified."""
    parts = table_name.split(".")

    if len(parts) != 3 or any(not part.strip() for part in parts):
        raise ValueError(
            "chunks_table must be fully qualified as "
            "'catalog.schema.table_name'. "
            f"Received: {table_name!r}"
        )


def _parsed_table_name(chunks_table: str) -> str:
    """Derive a parsed-document table name from the chunks table name."""
    catalog, schema, table = chunks_table.split(".")

    if table.endswith("_chunks"):
        table = table.removesuffix("_chunks")

    return f"{catalog}.{schema}.{table}_parsed_documents"


def build_chunks_table(
    spark: Any,
    volume_path: str,
    chunks_table: str,
) -> None:
    """Parse PDFs from a UC Volume and create the chunks Delta table.

    Parameters
    ----------
    spark:
        Active Databricks SparkSession.
    volume_path:
        Path to a PDF or directory in a Unity Catalog Volume.

        Examples:

        `/Volumes/main/default/pa4/annual_report.pdf`

        `/Volumes/main/default/pa4/`
    chunks_table:
        Fully qualified destination table name.

        Example:

        `main.default.pa4_chunks`

    Notes
    -----
    This function uses INSERT OVERWRITE so that it can safely be rerun during
    development without creating duplicate chunks.
    """
    if spark is None:
        raise ValueError(
            "A valid Databricks SparkSession must be supplied."
        )

    if not volume_path.startswith("/Volumes/"):
        raise ValueError(
            "volume_path must point to a Unity Catalog Volume, for example "
            "'/Volumes/main/default/pa4/annual_report.pdf'."
        )

    _validate_table_name(chunks_table)

    parsed_table = _parsed_table_name(chunks_table)

    print("=" * 70)
    print("Starting PA4 corpus ingestion")
    print(f"Source path:  {volume_path}")
    print(f"Parsed table: {parsed_table}")
    print(f"Chunks table: {chunks_table}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Parse PDF documents using ai_parse_document
    # ------------------------------------------------------------------

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {parsed_table} (
            source_path STRING,
            parsed_document VARIANT
        )
        USING DELTA
        TBLPROPERTIES (
            delta.enableChangeDataFeed = true
        )
        """
    )

    print("Parsing documents with ai_parse_document...")

    spark.sql(
        f"""
        INSERT OVERWRITE {parsed_table}
        SELECT
            path AS source_path,
            ai_parse_document(content) AS parsed_document
        FROM READ_FILES(
            '{volume_path}',
            format => 'binaryFile'
        )
        """
    )

    parsed_count = spark.table(parsed_table).count()

    if parsed_count == 0:
        raise RuntimeError(
            "No documents were parsed. Check that the PDF exists at "
            f"{volume_path!r} and that the current user has permission "
            "to read the Unity Catalog Volume."
        )

    print(f"Successfully parsed {parsed_count} document(s).")

    # ------------------------------------------------------------------
    # 2. Run ai_prep_search and expand the generated document chunks
    # ------------------------------------------------------------------

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {chunks_table} (
            chunk_id STRING,
            chunk_to_retrieve STRING,
            chunk_to_embed STRING,
            source STRING,
            page STRING
        )
        USING DELTA
        TBLPROPERTIES (
            delta.enableChangeDataFeed = true
        )
        """
    )

    print("Generating searchable chunks with ai_prep_search...")

    spark.sql(
        f"""
        INSERT OVERWRITE {chunks_table}
        SELECT
            chunk.value:chunk_id::STRING AS chunk_id,

            chunk.value:chunk_to_retrieve::STRING
                AS chunk_to_retrieve,

            COALESCE(
                chunk.value:chunk_to_embed::STRING,
                chunk.value:chunk_to_retrieve::STRING
            ) AS chunk_to_embed,

            regexp_extract(
                source_path,
                '[^/]+$',
                0
            ) AS source,

            COALESCE(
                chunk.value:metadata.page_number::STRING,
                chunk.value:metadata.page::STRING,
                chunk.value:page_number::STRING,
                chunk.value:page::STRING,
                'unknown'
            ) AS page

        FROM (
            SELECT
                source_path,
                ai_prep_search(parsed_document) AS prepared_document
            FROM {parsed_table}
        ) prepared,

        LATERAL variant_explode(
            prepared_document:document.contents
        ) AS chunk

        WHERE chunk.value:chunk_id::STRING IS NOT NULL
          AND chunk.value:chunk_to_retrieve::STRING IS NOT NULL
          AND LENGTH(
              TRIM(chunk.value:chunk_to_retrieve::STRING)
          ) > 0
        """
    )

    chunk_count = spark.table(chunks_table).count()

    if chunk_count == 0:
        raise RuntimeError(
            "ai_prep_search produced no chunks. Inspect the parsed table "
            f"{parsed_table!r} and confirm that ai_parse_document returned "
            "a valid document structure."
        )

    print(f"Successfully created {chunk_count} chunk(s).")

    # ------------------------------------------------------------------
    # 3. Verify required fields and primary-key uniqueness
    # ------------------------------------------------------------------

    missing_values = spark.sql(
        f"""
        SELECT COUNT(*) AS invalid_count
        FROM {chunks_table}
        WHERE chunk_id IS NULL
           OR chunk_to_retrieve IS NULL
           OR TRIM(chunk_to_retrieve) = ''
        """
    ).collect()[0]["invalid_count"]

    if missing_values:
        raise RuntimeError(
            f"The chunks table contains {missing_values} invalid row(s)."
        )

    duplicate_groups = spark.sql(
        f"""
        SELECT COUNT(*) AS duplicate_groups
        FROM (
            SELECT chunk_id
            FROM {chunks_table}
            GROUP BY chunk_id
            HAVING COUNT(*) > 1
        )
        """
    ).collect()[0]["duplicate_groups"]

    if duplicate_groups:
        raise RuntimeError(
            "The chunks table contains duplicate chunk_id values. "
            f"Duplicate groups found: {duplicate_groups}"
        )

    # Ensure CDF remains enabled even if the table existed previously.
    spark.sql(
        f"""
        ALTER TABLE {chunks_table}
        SET TBLPROPERTIES (
            delta.enableChangeDataFeed = true
        )
        """
    )

    print("\nChunk preview:")

    spark.sql(
        f"""
        SELECT
            chunk_id,
            LEFT(chunk_to_retrieve, 200) AS chunk_preview,
            source,
            page
        FROM {chunks_table}
        LIMIT 5
        """
    ).show(truncate=False)

    print("=" * 70)
    print("Delta chunks table created successfully.")
    print(f"Table: {chunks_table}")
    print(f"Rows:  {chunk_count}")
    print("=" * 70)


def _endpoint_exists(
    client: VectorSearchClient,
    endpoint_name: str,
) -> bool:
    """Return True when the Vector Search endpoint already exists."""
    response = client.list_endpoints()

    endpoints = response.get("endpoints", [])

    return any(
        endpoint.get("name") == endpoint_name
        for endpoint in endpoints
    )


def _index_exists(
    client: VectorSearchClient,
    endpoint_name: str,
    index_name: str,
) -> bool:
    """Return True when the index already exists on the endpoint."""
    try:
        response = client.list_indexes(
            endpoint_name=endpoint_name
        )
    except TypeError:
        # Some SDK versions use positional endpoint arguments.
        response = client.list_indexes(endpoint_name)

    indexes = (
        response.get("vector_indexes")
        or response.get("indexes")
        or []
    )

    return any(
        index.get("name") == index_name
        for index in indexes
    )


def _is_index_ready(description: dict[str, Any]) -> bool:
    """Check common Vector Search readiness response formats."""
    status = description.get("status", {})

    if status.get("ready") is True:
        return True

    possible_states = [
        status.get("detailed_state"),
        status.get("message"),
        status.get("state"),
        description.get("detailed_state"),
    ]

    return any(
        isinstance(state, str)
        and state.upper() in {
            "READY",
            "ONLINE",
            "PROVISIONED",
        }
        for state in possible_states
    )


def create_index(
    chunks_table: str | None = None,
    wait_timeout_seconds: int = 1800,
    poll_interval_seconds: int = 20,
):
    """Create the Vector Search endpoint and Delta Sync index.

    Parameters
    ----------
    chunks_table:
        Fully qualified source Delta table. When omitted, the function uses:

        `{UC_CATALOG}.{UC_SCHEMA}.pa4_chunks`
    wait_timeout_seconds:
        Maximum time to wait for the index to reach READY.
    poll_interval_seconds:
        Delay between status checks.

    Returns
    -------
    VectorSearchIndex
        Vector Search index handle.

    Required environment variables
    ------------------------------
    DATABRICKS_HOST
    DATABRICKS_TOKEN
    UC_CATALOG
    UC_SCHEMA
    VECTOR_SEARCH_ENDPOINT
    VECTOR_SEARCH_INDEX
    EMBEDDINGS_ENDPOINT
    """
    databricks_host = _get_required_env("DATABRICKS_HOST")
    databricks_token = _get_required_env("DATABRICKS_TOKEN")
    catalog = _get_required_env("UC_CATALOG")
    schema = _get_required_env("UC_SCHEMA")
    endpoint_name = _get_required_env("VECTOR_SEARCH_ENDPOINT")
    index_name = _get_required_env("VECTOR_SEARCH_INDEX")
    embedding_endpoint = _get_required_env("EMBEDDINGS_ENDPOINT")

    source_table = (
        chunks_table
        if chunks_table is not None
        else f"{catalog}.{schema}.pa4_chunks"
    )

    _validate_table_name(source_table)
    _validate_table_name(index_name)

    print("=" * 70)
    print("Creating or loading Databricks Vector Search resources")
    print(f"Endpoint:        {endpoint_name}")
    print(f"Index:           {index_name}")
    print(f"Source table:    {source_table}")
    print(f"Embedding model: {embedding_endpoint}")
    print("=" * 70)

    client = VectorSearchClient(
        workspace_url=databricks_host,
        personal_access_token=databricks_token,
        disable_notice=True,
    )

    # ------------------------------------------------------------------
    # 1. Create the STANDARD Vector Search endpoint
    # ------------------------------------------------------------------

    if not _endpoint_exists(client, endpoint_name):
        print(
            f"Creating STANDARD Vector Search endpoint "
            f"{endpoint_name!r}..."
        )

        client.create_endpoint(
            name=endpoint_name,
            endpoint_type="STANDARD",
        )

        print("Endpoint creation request submitted.")
    else:
        print(
            f"Vector Search endpoint {endpoint_name!r} "
            "already exists."
        )

    print("Waiting for the endpoint to become ready...")

    client.wait_for_endpoint(
        endpoint_name,
        timeout=timedelta(seconds=wait_timeout_seconds),
    )

    print("Vector Search endpoint is ready.")

    # ------------------------------------------------------------------
    # 2. Create the TRIGGERED Delta Sync index
    # ------------------------------------------------------------------

    if not _index_exists(
        client=client,
        endpoint_name=endpoint_name,
        index_name=index_name,
    ):
        print(f"Creating Delta Sync index {index_name!r}...")

        client.create_delta_sync_index(
            endpoint_name=endpoint_name,
            index_name=index_name,
            source_table_name=source_table,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_to_retrieve",
            embedding_model_endpoint_name=embedding_endpoint,
        )

        print("Index creation request submitted.")
    else:
        print(f"Vector Search index {index_name!r} already exists.")

    index = client.get_index(
        endpoint_name=endpoint_name,
        index_name=index_name,
    )

    # A TRIGGERED index needs an explicit sync after source-table changes.
    try:
        print("Triggering index synchronization...")
        index.sync()
    except Exception as exc:
        # A newly created index may already be performing its first sync.
        print(
            "An additional sync was not started. "
            "The index may already be synchronizing."
        )
        print(f"Sync response: {exc}")

    # ------------------------------------------------------------------
    # 3. Wait until the index reaches READY
    # ------------------------------------------------------------------

    deadline = time.time() + wait_timeout_seconds

    print("Waiting for the index to reach READY...")

    while time.time() < deadline:
        description = index.describe()

        if _is_index_ready(description):
            print("=" * 70)
            print("Vector Search index is READY.")
            print(f"Endpoint: {endpoint_name}")
            print(f"Index:    {index_name}")
            print("=" * 70)

            return index

        status = description.get("status", {})
        print(f"Current status: {status}")

        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Vector Search index {index_name!r} did not reach READY "
        f"within {wait_timeout_seconds} seconds."
    )


def test_similarity_search(
    index,
    query: str = (
        "What was Meridian Motor Corporation's "
        "net revenue in fiscal year 2023?"
    ),
    num_results: int = 4,
) -> dict[str, Any]:
    """Run a similarity-search test against the created index."""
    if num_results < 1:
        raise ValueError("num_results must be at least 1.")

    print("=" * 70)
    print("Running Vector Search similarity test")
    print(f"Query: {query}")
    print("=" * 70)

    result = index.similarity_search(
        query_text=query,
        columns=[
            "chunk_id",
            "chunk_to_retrieve",
            "source",
            "page",
        ],
        num_results=num_results,
    )

    rows = result.get("result", {}).get("data_array", [])

    if not rows:
        raise RuntimeError(
            "The similarity search returned no results. Confirm that:\n"
            "1. The chunks table contains data.\n"
            "2. The index has completed synchronization.\n"
            "3. The embedding endpoint is available.\n"
            "4. The index status is READY."
        )

    print(f"Similarity search returned {len(rows)} result(s).\n")

    for position, row in enumerate(rows, start=1):
        print(f"Result {position}:")
        print(row)
        print("-" * 70)

    return result