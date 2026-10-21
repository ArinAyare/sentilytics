"""Deploy the registered model to a managed online endpoint (and optionally delete it after the demo).

  python -m azure_ml.deploy_endpoint            # deploy
  python -m azure_ml.deploy_endpoint --delete   # STOP BILLING after your demo
"""
import argparse
import shutil
from pathlib import Path

from azure.ai.ml.entities import (CodeConfiguration, ManagedOnlineDeployment, ManagedOnlineEndpoint)

from azure_ml._client import get_client

ENDPOINT, DEPLOY, MODEL, ENV = "sentilytics-endpoint", "blue", "sentilytics-baseline", "sentilytics-env"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()
    ml = get_client()
    if a.delete:
        ml.online_endpoints.begin_delete(name=ENDPOINT).result()
        print("Endpoint deleted - billing stopped.")
        return

    shutil.copy("ml/common.py", "azure_ml/scoring/common.py")  # same preprocessing as training
    ml.online_endpoints.begin_create_or_update(
        ManagedOnlineEndpoint(name=ENDPOINT, auth_mode="key", description="Sentilytics sentiment API")).result()

    model = ml.models.get(MODEL, label="latest")
    env = ml.environments.get(ENV, label="latest")
    dep = ManagedOnlineDeployment(
        name=DEPLOY, endpoint_name=ENDPOINT, model=model, environment=env,
        code_configuration=CodeConfiguration(code="azure_ml/scoring", scoring_script="score.py"),
        instance_type="Standard_DS2_v2", instance_count=1)
    ml.online_deployments.begin_create_or_update(dep).result()

    ep = ml.online_endpoints.get(ENDPOINT)
    ep.traffic = {DEPLOY: 100}
    ml.online_endpoints.begin_create_or_update(ep).result()

    keys = ml.online_endpoints.get_keys(ENDPOINT)
    print("\nAdd these to your .env:")
    print(f"AZURE_ML_SCORING_URI={ep.scoring_uri}")
    print(f"AZURE_ML_ENDPOINT_KEY={keys.primary_key}")


if __name__ == "__main__":
    main()
