"""Metric computation + plots shared by every training script (identical JSON schema)."""
import json
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ml.common import LABELS


def compute_metrics(y_true, y_pred, model_name: str, extra: dict | None = None) -> dict:
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    out = {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro")), 4),
        "weighted_f1": round(float(rep["weighted avg"]["f1-score"]), 4),
        "per_class": {l: {k: round(float(rep[l][k]), 4) for k in ("precision", "recall", "f1-score")}
                      | {"support": int(rep[l]["support"])} for l in LABELS},
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "test_size": int(len(y_true)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        out.update(extra)
    return out


def save_metrics(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m = metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(m["matrix"], cmap="Blues")
        ax.set_xticks(range(3), m["labels"]); ax.set_yticks(range(3), m["labels"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f'{metrics["model_name"]} (acc={metrics["accuracy"]})')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, m["matrix"][i][j], ha="center", va="center")
        fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=150); plt.close(fig)
    except Exception as e:  # plotting must never break training
        print("plot skipped:", e)


def log_to_mlflow(metrics: dict, params: dict) -> None:
    """Inside an Azure ML job MLflow is pre-configured to log to the workspace automatically."""
    try:
        import mlflow
        mlflow.log_params(params)
        mlflow.log_metrics({k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1")})
    except Exception as e:
        print("mlflow logging skipped:", e)
