import os
from dataclasses import dataclass, field


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    predictor_mode: str = field(default_factory=lambda: os.getenv("PREDICTOR_MODE", "auto").lower())
    local_model_path: str = field(default_factory=lambda: os.getenv("LOCAL_MODEL_PATH", "artifacts/baseline/model.joblib"))
    artifacts_dir: str = field(default_factory=lambda: os.getenv("ARTIFACTS_DIR", "artifacts"))
    mongo_uri: str = field(default_factory=lambda: os.getenv("MONGO_URI", ""))
    mongo_db: str = field(default_factory=lambda: os.getenv("MONGO_DB", "sentilytics"))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    scoring_uri: str = field(default_factory=lambda: os.getenv("AZURE_ML_SCORING_URI", ""))
    endpoint_key: str = field(default_factory=lambda: os.getenv("AZURE_ML_ENDPOINT_KEY", ""))
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 5))
    max_batch_rows: int = field(default_factory=lambda: _int("MAX_BATCH_ROWS", 5000))
    cors_origins: list = field(default_factory=lambda: [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()])
