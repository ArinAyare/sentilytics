# Sentilytics
**Cloud-Based Sentiment Analysis and Feedback Analytics System Using Microsoft Azure Machine Learning**

Classifies text feedback as **positive / neutral / negative** with confidence scores. Trained and served via
**Azure ML** (jobs → MLflow tracking → model registry → managed online endpoint), consumed by a **FastAPI** backend
with **MongoDB**, and visualised in a **React** dashboard.

```
React (Vite) ──► FastAPI ──► Azure ML managed endpoint ──► sklearn / DistilBERT model
                    │            (auto-fallback to local model if unreachable)
                    └──► MongoDB (analyses, batch_jobs) ──► analytics / trends / CSV export
```

## Repo layout
| Path | Purpose |
|---|---|
| `ml/` | preprocessing (`common.py`), data prep, baseline (TF-IDF+LogReg), DistilBERT fine-tuning, shared evaluation |
| `azure_ml/` | Azure ML SDK v2 scripts: submit training job, register model, deploy/delete endpoint, `scoring/score.py` |
| `backend/app/` | FastAPI app, storage layer (Mongo / in-memory), predictor with Azure→local fallback |
| `frontend/` | React dashboard (Analyze, Batch CSV, Dashboard, History, Model Performance, ML Pipeline) |
| `tests/` | unit tests (preprocessing, services, API) |

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## 2. Get data and train locally (do this first - gives real numbers + the fallback model)
Recommended: **Twitter US Airline Sentiment** (Kaggle, `Tweets.csv`, native 3-class) → put in `ml/data/raw/`.
```bash
python -m ml.prepare_data --input ml/data/raw/Tweets.csv --text-col text --label-col airline_sentiment
python -m ml.train_baseline                  # -> artifacts/baseline/{model.joblib,metrics.json,confusion_matrix.png}
# optional, needs torch+transformers, ideally GPU (Azure ML NC4as_T4_v3 or Colab):
python -m ml.train_transformer               # -> artifacts/transformer/metrics.json
```
No dataset yet? `python -m ml.prepare_data --synthetic 6000` creates a **smoke-test-only** dataset (never report its numbers).

## 3. Run the app locally
```bash
PYTHONPATH=.:backend uvicorn app.main:app --reload       # API + Swagger at http://localhost:8000/docs
cd frontend && npm install && npm run dev                # UI at http://localhost:5173
```
Without `MONGO_URI` the API uses an in-memory store (fine for a quick demo; data resets on restart).
Tests: `pytest -q`.

## 4. Azure ML (the part examiners care about)
1. Azure Portal → create **Resource group** + **Azure Machine Learning workspace** (use Azure for Students credits).
2. `az login`, then set `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_ML_WORKSPACE` in your shell/.env.
3. Train in the cloud + register the model:
   `python -m azure_ml.submit_training_job`  → screenshot the **Job**, **Metrics** and **Models** pages in Azure ML Studio.
4. Deploy: `python -m azure_ml.deploy_endpoint` → prints `AZURE_ML_SCORING_URI` and `AZURE_ML_ENDPOINT_KEY`; paste into `.env`.
5. Restart the backend. The header badge now shows **Azure ML endpoint**; every prediction records `source: azure-ml`.
6. **After the demo/screenshots run `python -m azure_ml.deploy_endpoint --delete`** - endpoints bill hourly while running.

Direct test: `curl -X POST $AZURE_ML_SCORING_URI -H "Authorization: Bearer $AZURE_ML_ENDPOINT_KEY" -H "Content-Type: application/json" -d '{"texts":["great service"]}'`

## 5. Docker
```bash
docker compose up --build     # UI :8080, API :8000, MongoDB included  (train baseline first so artifacts/ exists)
```
Production deploy: push the two images to Azure Container Registry and run on Azure Container Apps / App Service; use
Azure Cosmos DB for MongoDB (or Atlas) via `MONGO_URI`; keep endpoint keys in Key Vault / app settings, never in git.

## API
`POST /api/predict` · `POST /api/batch` (CSV) · `GET /api/history` · `GET /api/analytics` · `GET /api/metrics` ·
`GET /api/jobs` · `GET /api/jobs/{id}/results` · `GET /api/jobs/{id}/download` · `GET /api/health`

## Security measures implemented
Input length validation · CSV-only + size/row limits · CSV formula-injection neutralisation on export · optional API key ·
restricted CORS · secrets only via environment variables · non-root Docker user · scoring script never leaks stack traces.
