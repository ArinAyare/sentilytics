"""Prediction back-ends: Azure ML endpoint (primary) and local joblib model (fallback)."""
import json
import logging

log = logging.getLogger("sentilytics.predictor")
CHUNK = 200


class AzurePredictor:
    name = "azure-ml"

    def __init__(self, scoring_uri: str, key: str, timeout: int = 30):
        self.uri, self.key, self.timeout = scoring_uri, key, timeout

    def predict(self, texts):
        import requests
        out = []
        for i in range(0, len(texts), CHUNK):
            r = requests.post(self.uri, json={"texts": texts[i:i + CHUNK]}, timeout=self.timeout,
                              headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            r.raise_for_status()
            body = r.json()
            if isinstance(body, str):  # some Azure ML versions double-encode
                body = json.loads(body)
            if "predictions" not in body:
                raise RuntimeError(f"Azure endpoint error: {body.get('error', 'unknown')}")
            out.extend(body["predictions"])
        return out


class LocalPredictor:
    name = "local-model"

    def __init__(self, path: str):
        import joblib
        self.model = joblib.load(path)

    def predict(self, texts):
        from ml.common import clean_text
        probs = self.model.predict_proba([clean_text(t) for t in texts])
        classes = list(self.model.classes_)
        res = []
        for row in probs:
            i = int(row.argmax())
            res.append({"label": classes[i], "confidence": round(float(row[i]), 4),
                        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, row)}})
        return res


class FallbackPredictor:
    """Try Azure first; if it is unreachable, serve from the local model so the demo never dies."""

    def __init__(self, primary, secondary):
        self.primary, self.secondary = primary, secondary

    def predict_with_source(self, texts):
        if self.primary is not None:
            try:
                return self.primary.predict(texts), self.primary.name
            except Exception as e:
                log.warning("Azure endpoint failed (%s); falling back to local model", e)
        if self.secondary is None:
            raise RuntimeError("No prediction backend available")
        return self.secondary.predict(texts), self.secondary.name


def build_predictor(s):
    azure = AzurePredictor(s.scoring_uri, s.endpoint_key) if s.scoring_uri and s.endpoint_key else None
    local = None
    if s.predictor_mode in ("local", "auto"):
        try:
            local = LocalPredictor(s.local_model_path)
        except Exception as e:
            log.warning("Local model unavailable: %s", e)
    if s.predictor_mode == "azure":
        local = None
    if s.predictor_mode == "local":
        azure = None
    return FallbackPredictor(azure, local)
