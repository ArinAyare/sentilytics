"""Azure ML managed online endpoint scoring script.
Request : {"texts": ["great product", "worst ever"]}
Response: {"predictions": [{"label": "...", "confidence": 0.93, "probabilities": {"negative":..,"neutral":..,"positive":..}}]}
"""
import json
import logging
import os
from pathlib import Path

import joblib

from common import clean_text  # copied next to this file by deploy_endpoint.py

MODEL = None
MAX_TEXTS, MAX_CHARS = 500, 5000


def init():
    global MODEL
    root = Path(os.environ["AZUREML_MODEL_DIR"])
    path = next(root.rglob("model.joblib"))
    MODEL = joblib.load(path)
    logging.info("Model loaded from %s", path)


def run(raw_data):
    try:
        texts = json.loads(raw_data)["texts"]
        if not isinstance(texts, list) or not texts or len(texts) > MAX_TEXTS:
            return {"error": f"'texts' must be a list of 1..{MAX_TEXTS} strings"}
        cleaned = [clean_text(str(t)[:MAX_CHARS]) for t in texts]
        probs = MODEL.predict_proba(cleaned)
        classes = list(MODEL.classes_)
        out = []
        for row in probs:
            i = int(row.argmax())
            out.append({"label": classes[i], "confidence": round(float(row[i]), 4),
                        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, row)}})
        return {"predictions": out}
    except Exception as e:  # never leak stack traces to callers
        logging.exception("scoring failed")
        return {"error": f"bad request: {type(e).__name__}"}
