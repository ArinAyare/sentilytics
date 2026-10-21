import sys
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure_ml._client import get_client

job_name = sys.argv[1]
ml = get_client()
m = ml.models.create_or_update(Model(
    name="sentilytics-baseline", type=AssetTypes.CUSTOM_MODEL,
    path=f"azureml://jobs/{job_name}/outputs/model_out",
    description="TF-IDF + Logistic Regression sentiment classifier"))
print(f"Registered {m.name} v{m.version}")
