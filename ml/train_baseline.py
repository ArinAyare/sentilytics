"""Baseline: TF-IDF (1-2 grams) + Logistic Regression. Fast, strong, and cheap to host on Azure ML."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.evaluate import compute_metrics, log_to_mlflow, save_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/processed")
    ap.add_argument("--out-dir", default="artifacts/baseline")
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--max-features", type=int, default=100_000)
    a = ap.parse_args()

    d = Path(a.data_dir)
    train, val, test = (pd.read_csv(d / f"{n}.csv") for n in ("train", "val", "test"))
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=a.max_features)),
        ("clf", LogisticRegression(C=a.C, max_iter=1000, class_weight="balanced")),
    ])
    model.fit(train["text"], train["label"])
    val_acc = float((model.predict(val["text"]) == val["label"]).mean())
    pred = model.predict(test["text"])
    metrics = compute_metrics(test["label"], pred, "TF-IDF + Logistic Regression",
                              {"train_size": len(train), "val_accuracy": round(val_acc, 4),
                               "hyperparameters": {"C": a.C, "ngram_range": [1, 2], "max_features": a.max_features}})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    save_metrics(metrics, out)
    log_to_mlflow(metrics, {"model": "tfidf_logreg", "C": a.C, "max_features": a.max_features})
    print(f"accuracy={metrics['accuracy']} macro_f1={metrics['macro_f1']} -> {out}")


if __name__ == "__main__":
    main()
