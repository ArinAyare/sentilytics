"""Fine-tune DistilBERT for 3-class sentiment (needs: pip install torch transformers).
Run on an Azure ML GPU compute (e.g. Standard_NC4as_T4_v3) or Colab. Outputs the same metrics.json schema
as the baseline so the dashboard can compare both models."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments)

from ml.common import LABELS
from ml.evaluate import compute_metrics, log_to_mlflow, save_metrics

L2I = {l: i for i, l in enumerate(LABELS)}


class DS(torch.utils.data.Dataset):
    def __init__(self, df, tok, max_len):
        self.enc = tok(df["text"].tolist(), truncation=True, padding=True, max_length=max_len)
        self.y = df["label"].map(L2I).tolist()

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.y[i])
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/processed")
    ap.add_argument("--out-dir", default="artifacts/transformer")
    ap.add_argument("--model-name", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--max-len", type=int, default=128)
    a = ap.parse_args()

    d = Path(a.data_dir)
    train, val, test = (pd.read_csv(d / f"{n}.csv") for n in ("train", "val", "test"))
    tok = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model_name, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=L2I)
    args = TrainingArguments(output_dir=str(Path(a.out_dir) / "ckpt"), num_train_epochs=a.epochs,
                             per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=64,
                             learning_rate=a.lr, weight_decay=0.01, eval_strategy="epoch",
                             save_strategy="no", logging_steps=50, report_to="none")
    tr = Trainer(model=model, args=args, train_dataset=DS(train, tok, a.max_len), eval_dataset=DS(val, tok, a.max_len))
    tr.train()
    logits = tr.predict(DS(test, tok, a.max_len)).predictions
    pred = [LABELS[i] for i in np.argmax(logits, axis=1)]
    metrics = compute_metrics(test["label"], pred, f"Fine-tuned {a.model_name}",
                              {"train_size": len(train), "hyperparameters": {"epochs": a.epochs, "lr": a.lr,
                                                                              "batch_size": a.batch_size, "max_len": a.max_len}})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "model"); tok.save_pretrained(out / "model")
    save_metrics(metrics, out)
    log_to_mlflow(metrics, {"model": a.model_name, "epochs": a.epochs, "lr": a.lr})
    print(f"accuracy={metrics['accuracy']} macro_f1={metrics['macro_f1']}")


if __name__ == "__main__":
    main()
