"""Build cleaned, stratified train/val/test CSVs.

Examples
  # Twitter US Airline Sentiment (Kaggle 'Tweets.csv'): native 3-class labels
  python -m ml.prepare_data --input ml/data/raw/Tweets.csv --text-col text --label-col airline_sentiment
  # Star-rated reviews (1-2 neg, 3 neu, 4-5 pos)
  python -m ml.prepare_data --input reviews.csv --text-col reviewText --label-col overall --stars
  # Smoke-test dataset ONLY (synthetic - do not report these numbers as research results)
  python -m ml.prepare_data --synthetic 6000
"""
import argparse
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.common import LABELS, clean_text


def map_label(v, stars: bool):
    s = str(v).strip().lower()
    if stars:
        try:
            r = float(s)
        except ValueError:
            return None
        return "negative" if r <= 2 else "neutral" if r == 3 else "positive"
    if s in LABELS:
        return s
    return {"0": "negative", "1": "neutral", "2": "positive", "-1": "negative",
            "neg": "negative", "neu": "neutral", "pos": "positive"}.get(s)


def synthetic(n: int, seed: int = 42) -> pd.DataFrame:
    rnd = random.Random(seed)
    subj = ["the product", "the service", "the app", "delivery", "the staff", "this course", "the food", "support"]
    pos = ["is excellent", "was amazing", "works perfectly", "exceeded my expectations", "is really great", "was very helpful"]
    neg = ["is terrible", "was awful", "keeps crashing", "was a complete waste of money", "is really disappointing", "was very rude"]
    neu = ["is okay", "was average", "arrived on Tuesday", "is what the description says", "is neither good nor bad", "was fine I guess"]
    extra = ["", " Would recommend!", " Never again.", " Not sure what else to say.", " Overall.", " Honestly."]
    rows = []
    for _ in range(n):
        lab = rnd.choice(LABELS)
        pool = {"positive": pos, "negative": neg, "neutral": neu}[lab]
        txt = f"{rnd.choice(subj).capitalize()} {rnd.choice(pool)}.{rnd.choice(extra)}"
        if rnd.random() < 0.08:  # label noise so the task is not trivial
            lab = rnd.choice(LABELS)
        rows.append((txt, lab))
    return pd.DataFrame(rows, columns=["text", "label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input"); ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label"); ap.add_argument("--stars", action="store_true")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--out-dir", default="ml/data/processed"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.synthetic:
        df = synthetic(a.synthetic, a.seed)
    else:
        if not a.input:
            ap.error("--input is required unless --synthetic is used")
        raw = pd.read_csv(a.input)
        df = pd.DataFrame({"text": raw[a.text_col], "label": raw[a.label_col].map(lambda v: map_label(v, a.stars))})
    n0 = len(df)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() >= 3].drop_duplicates(subset="text").reset_index(drop=True)
    print(f"rows: {n0} -> {len(df)} after cleaning/dedup\n{df['label'].value_counts().to_string()}")

    train, tmp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=a.seed)
    val, test = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(out / f"{name}.csv", index=False)
    print(f"saved train={len(train)} val={len(val)} test={len(test)} -> {out}")


if __name__ == "__main__":
    main()
