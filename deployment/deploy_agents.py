"""Bonus B — deploy via the databricks-agents SDK (deployment/deploy_agents.py).

TODO: Log + register the model (reuse the pattern from deploy.py), then call
`databricks.agents.deploy(model_name=..., model_version=...)` to provision the
serving endpoint AND the Review App in one call. Print the endpoint + review URL.
"""

from __future__ import annotations

import os
from importlib.metadata import version as package_version
from pathlib import Path

import mlflow
from databricks import agents
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

    return value


def installed_version(package: str) -> str:
    return package_version(package)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    os.chdir(PROJECT_ROOT)

    catalog = require_env("UC_CATALOG")
    schema = require_env("UC_SCHEMA")

    require_env("DATABRICKS_HOST")
    require_env("DATABRICKS_TOKEN")
    require_env("DATABRICKS_MODEL")
    require_env("VECTOR_SEARCH_ENDPOINT")
    require_env("VECTOR_SEARCH_INDEX")
    require_env("EMBEDDINGS_ENDPOINT")

    model_name = os.getenv(
        "DOCUMENT_ANALYST_V2_MODEL_NAME",
        "pa4_document_analyst_v2",
    ).strip()

    uc_model_name = (
        f"{catalog}.{schema}.{model_name}"
    )

    experiment_name = os.getenv(
        "BONUS_B_EXPERIMENT_NAME",
        "/Shared/pa4-document-analyst-bonus-b",
    ).strip()

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_name)

    pip_requirements = [
        f"mlflow=={installed_version('mlflow')}",
        (
            "databricks-agents=="
            f"{installed_version('databricks-agents')}"
        ),
        (
            "databricks-langchain=="
            f"{installed_version('databricks-langchain')}"
        ),
        (
            "databricks-vectorsearch=="
            f"{installed_version('databricks-vectorsearch')}"
        ),
        f"langgraph=={installed_version('langgraph')}",
        f"langchain=={installed_version('langchain')}",
        (
            "langchain-core=="
            f"{installed_version('langchain-core')}"
        ),
        (
            "langchain-openai=="
            f"{installed_version('langchain-openai')}"
        ),
        (
            "langchain-mcp-adapters=="
            f"{installed_version('langchain-mcp-adapters')}"
        ),
        f"mcp=={installed_version('mcp')}",
        f"openai=={installed_version('openai')}",
        "python-dotenv",
    ]

    print("=" * 72)
    print("BONUS B: Logging Agent Framework model")
    print(f"Experiment: {experiment_name}")
    print(f"UC model:   {uc_model_name}")
    print("=" * 72)


    os.chdir(PROJECT_ROOT)

    print("Current directory:", os.getcwd())
    print(
        "Agent file exists:",
        os.path.exists("deployment/agent_chat.py"),
    )
    with mlflow.start_run(
        run_name="pa4-document-analyst-v2"
    ) as run:
        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model="deployment/agent_chat.py",
            code_paths=[
                "agent",
                "rag",
                "tools",
                "config.py",
            ],
            input_example={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "What was Meridian's revenue in 2023, "
                            "and what would it be after a 10% increase?"
                        ),
                    }
                ]
            },
            pip_requirements=pip_requirements,
        )

        run_id = run.info.run_id

    print("Model URI:", model_info.model_uri)
    print("Run ID:", run_id)

    registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=uc_model_name,
    )

    version = str(registered.version)

    print("=" * 72)
    print("Model registered")
    print(f"Name:    {uc_model_name}")
    print(f"Version: {version}")
    print("=" * 72)

    deployment = agents.deploy(
        model_name=uc_model_name,
        model_version=version,
        scale_to_zero=True,
        environment_vars={
            "DATABRICKS_MODEL": require_env(
                "DATABRICKS_MODEL"
            ),
            "VECTOR_SEARCH_ENDPOINT": require_env(
                "VECTOR_SEARCH_ENDPOINT"
            ),
            "VECTOR_SEARCH_INDEX": require_env(
                "VECTOR_SEARCH_INDEX"
            ),
            "EMBEDDINGS_ENDPOINT": require_env(
                "EMBEDDINGS_ENDPOINT"
            ),
        },
    )

    print("=" * 72)
    print("Bonus B deployment submitted")
    print(f"Endpoint:   {deployment.endpoint_name}")
    print(f"Review App: {deployment.review_app_url}")
    print("=" * 72)


if __name__ == "__main__":
    main()