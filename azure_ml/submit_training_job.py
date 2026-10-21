"""Submit the baseline training as an Azure ML command job, then register the model.

Prereqs: `az login`, an Azure ML workspace, ml/data/processed/{train,val,test}.csv exist locally.
Run:  python -m azure_ml.submit_training_job
"""
from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Data, Environment, Model

from azure_ml._client import get_client

COMPUTE, ENV_NAME, MODEL_NAME = "cpu-cluster", "sentilytics-env", "sentilytics-baseline"


def main():
    ml = get_client()

    try:
        ml.compute.get(COMPUTE)
    except Exception:
        print("Creating compute cluster (scales to 0 when idle)...")
        ml.compute.begin_create_or_update(AmlCompute(name=COMPUTE, size="Standard_DS2_v2",
                                                     min_instances=0, max_instances=1,
                                                     idle_time_before_scale_down=120)).result()

    data = ml.data.create_or_update(Data(name="sentilytics-dataset", path="ml/data/processed",
                                         type=AssetTypes.URI_FOLDER, description="Cleaned train/val/test CSVs"))
    env = ml.environments.create_or_update(Environment(
        name=ENV_NAME, conda_file="azure_ml/conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"))

    job = command(
        code=".",  # repo root (honours .amlignore if present)
        command=("python -m ml.train_baseline --data-dir ${{inputs.data}} --out-dir ${{outputs.model_out}}"),
        inputs={"data": Input(type=AssetTypes.URI_FOLDER, path=data.id)},
        outputs={"model_out": Output(type=AssetTypes.URI_FOLDER)},
        environment=f"{env.name}:{env.version}", compute=COMPUTE,
        experiment_name="sentilytics-sentiment", display_name="baseline-tfidf-logreg")
    job = ml.jobs.create_or_update(job)
    print("Submitted:", job.studio_url)
    ml.jobs.stream(job.name)

    model = ml.models.create_or_update(Model(
        name=MODEL_NAME, type=AssetTypes.CUSTOM_MODEL,
        path=f"azureml://jobs/{job.name}/outputs/model_out",
        description="TF-IDF + Logistic Regression sentiment classifier"))
    print(f"Registered model {model.name} v{model.version}")


if __name__ == "__main__":
    main()
