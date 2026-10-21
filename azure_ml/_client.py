import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def get_client() -> MLClient:
    missing = [k for k in ("AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ML_WORKSPACE") if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Set these environment variables first: {', '.join(missing)}")
    return MLClient(DefaultAzureCredential(), os.environ["AZURE_SUBSCRIPTION_ID"],
                    os.environ["AZURE_RESOURCE_GROUP"], os.environ["AZURE_ML_WORKSPACE"])
