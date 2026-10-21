"""Business logic, independent of FastAPI so it can be unit-tested directly."""
import csv
import io

import pandas as pd

from app.repository import LABELS, new_id, now

MAX_TEXT_CHARS = 5000
TEXT_COLUMN_HINTS = ["text", "review", "reviews", "feedback", "comment", "comments", "content", "message"]


class ValidationError(ValueError):
    pass


def validate_text(text) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValidationError("Text must not be empty.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValidationError(f"Text too long (max {MAX_TEXT_CHARS} characters).")
    return text.strip()


def analyze_text(text, repo, predictor):
    text = validate_text(text)
    preds, source = predictor.predict_with_source([text])
    p = preds[0]
    doc = {"id": new_id(), "text": text, "label": p["label"], "confidence": p["confidence"],
           "probabilities": p["probabilities"], "source": source, "job_id": None, "row": 0, "created_at": now()}
    repo.add_analyses([doc])
    return doc


def _read_csv(data: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            raise ValidationError(f"Could not parse CSV: {e}")
    raise ValidationError("Unsupported file encoding.")


def detect_text_column(df, requested=None):
    if requested:
        if requested not in df.columns:
            raise ValidationError(f"Column '{requested}' not found. Available: {list(df.columns)}")
        return requested
    lower = {c.lower().strip(): c for c in df.columns}
    for h in TEXT_COLUMN_HINTS:
        if h in lower:
            return lower[h]
    # fall back to column with the longest average string length
    if len(df.columns):
        return max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())
    raise ValidationError("CSV has no columns.")


def analyze_csv(data: bytes, filename, repo, predictor, max_rows, text_column=None):
    df = _read_csv(data)
    if df.empty:
        raise ValidationError("CSV contains no rows.")
    col = detect_text_column(df, text_column)
    texts = [str(t).strip()[:MAX_TEXT_CHARS] for t in df[col]]
    texts = [t for t in texts if t]
    if not texts:
        raise ValidationError(f"Column '{col}' has no non-empty text.")
    truncated = len(texts) > max_rows
    texts = texts[:max_rows]

    preds, source = predictor.predict_with_source(texts)
    job_id, ts = new_id(), now()
    docs = [{"id": new_id(), "text": t, "label": p["label"], "confidence": p["confidence"],
             "probabilities": p["probabilities"], "source": source, "job_id": job_id, "row": i, "created_at": ts}
            for i, (t, p) in enumerate(zip(texts, preds))]
    repo.add_analyses(docs)
    counts = {l: sum(1 for d in docs if d["label"] == l) for l in LABELS}
    job = {"id": job_id, "filename": filename, "text_column": col, "total_rows": len(docs), "counts": counts,
           "avg_confidence": round(sum(d["confidence"] for d in docs) / len(docs), 4),
           "source": source, "truncated": truncated, "created_at": ts}
    repo.add_job(job)
    return job


def _safe_cell(v):
    """Neutralise CSV/Excel formula injection."""
    s = str(v)
    return "'" + s if s and s[0] in "=+-@\t\r" else s


def export_csv(job_id, repo) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["text", "sentiment", "confidence", "prob_negative", "prob_neutral", "prob_positive"])
    skip = 0
    while True:
        batch = repo.list_analyses(skip=skip, limit=1000, job_id=job_id)
        if not batch:
            break
        for d in batch:
            pr = d["probabilities"]
            w.writerow([_safe_cell(d["text"]), d["label"], d["confidence"],
                        pr.get("negative", 0), pr.get("neutral", 0), pr.get("positive", 0)])
        skip += 1000
    return buf.getvalue()


def analytics(repo):
    counts = repo.label_counts()
    total = sum(counts.values())
    pct = {l: round(100 * counts[l] / total, 1) if total else 0 for l in LABELS}
    return {"total": total, "counts": counts, "percentages": pct,
            "avg_confidence": round(repo.avg_confidence(), 4), "trend": repo.daily_trend(30)}
