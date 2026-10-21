"""Sentilytics REST API.  Run from repo root:  PYTHONPATH=.:backend uvicorn app.main:app --reload
Interactive docs: http://localhost:8000/docs"""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app import services
from app.config import Settings
from app.predictor import build_predictor
from app.repository import build_repo

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sentilytics")
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.repo = build_repo(settings)
    app.state.predictor = build_predictor(settings)
    log.info("storage=%s azure=%s local=%s", app.state.repo.kind,
             bool(app.state.predictor.primary), bool(app.state.predictor.secondary))
    yield


app = FastAPI(title="Sentilytics API", version="1.0.0",
              description="Cloud-based sentiment analysis using Azure Machine Learning", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"])


def require_key(x_api_key: str | None = Header(default=None)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(401, "Invalid or missing API key")


def _ser(doc: dict) -> dict:
    d = {k: v for k, v in doc.items() if k != "_id"}
    if "created_at" in d:
        d["created_at"] = d["created_at"].isoformat()
    return d


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=services.MAX_TEXT_CHARS)


def _guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except services.ValidationError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        log.error("prediction backend failure: %s", e)
        raise HTTPException(503, "No prediction backend available")


@app.get("/api/health")
def health():
    p = app.state.predictor
    return {"status": "ok", "storage": app.state.repo.kind,
            "azure_endpoint_configured": p.primary is not None, "local_fallback_loaded": p.secondary is not None}


@app.post("/api/predict", dependencies=[Depends(require_key)])
def predict(body: PredictIn):
    return _ser(_guard(services.analyze_text, body.text, app.state.repo, app.state.predictor))


@app.post("/api/batch", dependencies=[Depends(require_key)])
async def batch(file: UploadFile = File(...), text_column: str | None = Form(default=None)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "Only .csv files are accepted")
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
    job = _guard(services.analyze_csv, data, file.filename, app.state.repo, app.state.predictor,
                 settings.max_batch_rows, text_column or None)
    return _ser(job)


@app.get("/api/jobs", dependencies=[Depends(require_key)])
def jobs(limit: int = Query(20, ge=1, le=100)):
    return [_ser(j) for j in app.state.repo.list_jobs(limit)]


@app.get("/api/jobs/{job_id}/results", dependencies=[Depends(require_key)])
def job_results(job_id: str, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)):
    if not app.state.repo.get_job(job_id):
        raise HTTPException(404, "Job not found")
    repo = app.state.repo
    items = repo.list_analyses(skip=(page - 1) * page_size, limit=page_size, job_id=job_id)
    return {"total": repo.count(job_id=job_id), "page": page, "items": [_ser(i) for i in items]}


@app.get("/api/jobs/{job_id}/download", dependencies=[Depends(require_key)])
def download(job_id: str):
    if not app.state.repo.get_job(job_id):
        raise HTTPException(404, "Job not found")
    content = services.export_csv(job_id, app.state.repo)
    return Response(content, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="sentilytics_{job_id[:8]}.csv"'})


@app.get("/api/history", dependencies=[Depends(require_key)])
def history(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
            label: str | None = Query(None, pattern="^(positive|neutral|negative)$")):
    repo = app.state.repo
    items = repo.list_analyses(skip=(page - 1) * page_size, limit=page_size, label=label)
    return {"total": repo.count(label=label), "page": page, "page_size": page_size, "items": [_ser(i) for i in items]}


@app.get("/api/analytics", dependencies=[Depends(require_key)])
def analytics():
    return services.analytics(app.state.repo)


@app.get("/api/metrics")
def metrics():
    """Returns metrics.json of every trained model found under ARTIFACTS_DIR (baseline, transformer, ...)."""
    out = []
    for f in sorted(Path(settings.artifacts_dir).glob("*/metrics.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            log.warning("unreadable metrics file %s", f)
    return out
