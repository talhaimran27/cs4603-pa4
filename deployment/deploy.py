"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py

TODO:
  - `log_and_register()`: set registry uri to 'databricks-uc', log the model via
    `mlflow.langchain.log_model(lc_model="deployment/agent_model.py", name=...,
    code_paths=[...], pip_requirements=[...], input_example={...})`, then
    `mlflow.register_model(...)` into $UC_CATALOG.$UC_SCHEMA.<model>.
  - `create_or_update_endpoint(uc_name, version)`: create/update a Model Serving
    endpoint with `WorkspaceClient().serving_endpoints`, workload_size='Small',
    scale_to_zero_enabled=True, and environment_vars supplied as secret refs
    ({{secrets/cs4603-deploy/...}}). Wait for READY and print the URL.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
import mlflow
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)


DEPLOYMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOYMENT_DIR.parent
AGENT_MODEL_PATH = DEPLOYMENT_DIR / "agent_model.py"


# Packages required inside the Databricks serving container.
PIP_REQUIREMENTS = [
    "mlflow",
    "langgraph",
    "langchain",
    "langchain-core",
    "langchain-openai",
    "databricks-langchain",
    "databricks-vectorsearch",
    "databricks-sdk",
    "langchain-mcp-adapters",
    "mcp",
    "openai",
    "python-dotenv",
]


def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Add it to your .env file before running deployment/deploy.py."
        )

    return value

def validate_local_files() -> None:
    """Ensure all files that MLflow must package exist."""
    required_paths = [
        AGENT_MODEL_PATH,
        PROJECT_ROOT / "agent",
        PROJECT_ROOT / "rag",
        PROJECT_ROOT / "tools",
        PROJECT_ROOT / "config.py",
    ]

    missing = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required deployment files are missing:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

def log_and_register() -> tuple[str, str]:
    """Log the model to MLflow and register it in Unity Catalog."""

    load_dotenv(PROJECT_ROOT / ".env")
    os.chdir(PROJECT_ROOT)
    validate_local_files()

    catalog = require_env("UC_CATALOG")
    schema = require_env("UC_SCHEMA")

    # These variables are needed when MLflow imports agent_model.py.
    require_env("DATABRICKS_HOST")
    require_env("DATABRICKS_TOKEN")
    require_env("DATABRICKS_MODEL")
    require_env("VECTOR_SEARCH_ENDPOINT")
    require_env("VECTOR_SEARCH_INDEX")
    require_env("EMBEDDINGS_ENDPOINT")

    model_name = os.getenv(
        "DOCUMENT_ANALYST_MODEL_NAME",
        "pa4_document_analyst",
    ).strip()

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "/Shared/pa4-document-analyst",
    ).strip()

    uc_model_name = f"{catalog}.{schema}.{model_name}"

    # Use Databricks MLflow and Unity Catalog.
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_name)

    code_paths=[
    "agent",
    "rag",
    "tools",
    "config.py",
    ]

    input_example = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "What was Meridian's revenue in 2023, and what would "
                    "it be after a 10 percent increase?"
                ),
            }
        ]
    }

    print("=" * 72)
    print("Logging Document Analyst")
    print(f"Experiment:    {experiment_name}")
    print(f"Model file:    {AGENT_MODEL_PATH}")
    print(f"UC model name: {uc_model_name}")
    print("=" * 72)

    with mlflow.start_run(
        run_name="pa4-document-analyst"
    ) as run:
        model_info = mlflow.langchain.log_model(
            lc_model="deployment/agent_model.py",
            name="agent",
            code_paths=code_paths,
            pip_requirements=PIP_REQUIREMENTS,
            input_example=input_example,
        )

        run_id = run.info.run_id

    print(f"MLflow run ID:   {run_id}")
    print(f"Logged model URI: {model_info.model_uri}")

    registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=uc_model_name,
    )

    version = str(registered.version)

    print("=" * 72)
    print("Model registered successfully")
    print(f"Registered model: {uc_model_name}")
    print(f"Registered version: {version}")
    print("=" * 72)

    return uc_model_name, version
    

def _enum_value(value) -> str:
    return getattr(value, "value", str(value).split(".")[-1])


def wait_for_endpoint_ready(
    workspace,
    endpoint_name: str,
    timeout_seconds: int = 1200,
    poll_seconds: int = 20,
) -> None:
    import time

    started = time.monotonic()

    while True:
        endpoint = workspace.serving_endpoints.get(endpoint_name)

        ready_state = _enum_value(endpoint.state.ready)
        update_state = _enum_value(
            endpoint.state.config_update
        )

        elapsed = int(time.monotonic() - started)

        print(
            f"Endpoint status after {elapsed}s: "
            f"ready={ready_state}, "
            f"config_update={update_state}"
        )

        if (
            ready_state == "READY"
            and update_state == "NOT_UPDATING"
        ):
            print("Endpoint is READY.")
            return

        if update_state == "UPDATE_FAILED":
            raise RuntimeError(
                f"Endpoint {endpoint_name!r} deployment failed."
            )

        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Endpoint {endpoint_name!r} did not become READY."
            )

        time.sleep(poll_seconds)



def create_or_update_endpoint(uc_name: str, version: str) -> str:
    load_dotenv(PROJECT_ROOT / ".env")

    host = require_env("DATABRICKS_HOST").rstrip("/")

    vector_search_endpoint = require_env(
        "VECTOR_SEARCH_ENDPOINT"
    )
    vector_search_index = require_env(
        "VECTOR_SEARCH_INDEX"
    )
    embeddings_endpoint = require_env(
        "EMBEDDINGS_ENDPOINT"
    )

    endpoint_name = require_env("SERVING_ENDPOINT_NAME")
    secret_scope = require_env("SECRET_SCOPE")

    workspace = WorkspaceClient()

    environment_vars = {
        "DATABRICKS_HOST": (
            f"{{{{secrets/{secret_scope}/DATABRICKS_HOST}}}}"
        ),
        "DATABRICKS_TOKEN": (
            f"{{{{secrets/{secret_scope}/DATABRICKS_TOKEN}}}}"
        ),
        "DATABRICKS_MODEL": (
            f"{{{{secrets/{secret_scope}/DATABRICKS_MODEL}}}}"
        ),
        "VECTOR_SEARCH_ENDPOINT": vector_search_endpoint,
        "VECTOR_SEARCH_INDEX": vector_search_index,
        "EMBEDDINGS_ENDPOINT": embeddings_endpoint,
    }

    served_entity = ServedEntityInput(
        entity_name=uc_name,
        entity_version=str(version),
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars=environment_vars,
    )

    print("=" * 72)
    print("Deploying Document Analyst endpoint")
    print(f"Endpoint name: {endpoint_name}")
    print(f"UC model:      {uc_name}")
    print(f"Model version: {version}")
    print("=" * 72)

    try:
        existing = workspace.serving_endpoints.get(endpoint_name)
    except Exception as exc:
        # The SDK may raise an error when the endpoint does not exist.
        error_text = str(exc).lower()

        if (
            "does not exist" not in error_text
            and "resource_does_not_exist" not in error_text
            and "not found" not in error_text
        ):
            raise

        existing = None

    if existing is None:
        print("Endpoint does not exist. Creating it...")

        config = EndpointCoreConfigInput(
            name=endpoint_name,
            served_entities=[served_entity],
        )

        workspace.serving_endpoints.create(
            name=endpoint_name,
            config=config,
        )

        print("Endpoint creation request submitted.")

    else:
        print("Endpoint already exists.")

        existing_state = existing.state
        update_state = str(
            getattr(existing_state, "config_update", "")
        ).upper()

        if "IN_PROGRESS" in update_state:
            print(
                "An endpoint configuration update is already in progress. "
                "Waiting for it to finish before submitting the new version..."
            )

            wait_for_endpoint_ready(
                workspace=workspace,
                endpoint_name=endpoint_name,
            )

        print(
            f"Updating endpoint to {uc_name} version {version}..."
        )

        workspace.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=[served_entity],
        )

        print("Endpoint update request submitted.")

    wait_for_endpoint_ready(
        workspace=workspace,
        endpoint_name=endpoint_name,
    )

    endpoint_url = (
        f"{host}/serving-endpoints/"
        f"{endpoint_name}/invocations"
    )

    print("=" * 72)
    print("Deployment completed successfully")
    print(f"Endpoint: {endpoint_name}")
    print(f"URL:      {endpoint_url}")
    print("=" * 72)

    return endpoint_url


if __name__ == "__main__":
    # name, ver = log_and_register()
    # create_or_update_endpoint(name, ver)
    create_or_update_endpoint(
        "main.default.pa4_document_analyst",
        "6",
    )
