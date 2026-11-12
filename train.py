"""Train and evaluate the TF-IDF + logistic regression classifier.

Usage: python train.py
Writes model/model.joblib and metrics.json.
"""
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).parent
DATA = ROOT / "assets" / "Dataset"
SEED = 42
TEST_SIZE = 0.2
FAKE_LABEL = 1  # 1 = fake, 0 = true

# Source tells in the True set: "WASHINGTON (Reuters) - ..." datelines and the word Reuters.
REUTERS_CUE = re.compile(r"^[^()\-]{0,80}\(Reuters\)\s*-?\s*|\bReuters\b", re.IGNORECASE)


def load() -> pd.DataFrame:
    fake = pd.read_csv(DATA / "Fake.csv").assign(label=FAKE_LABEL)
    true = pd.read_csv(DATA / "True.csv").assign(label=0)
    df = pd.concat([fake, true], ignore_index=True)[["text", "label"]]
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    return df.drop_duplicates(subset="text")  # duplicates would leak across the split


def build():
    return make_pipeline(
        TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2),
        LogisticRegression(max_iter=1000),
    )


def evaluate(df: pd.DataFrame) -> dict:
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"])
    model = build().fit(train["text"], train["label"])
    pred = model.predict(test["text"])
    p, r, f, _ = precision_recall_fscore_support(test["label"], pred, labels=[FAKE_LABEL], zero_division=0)
    return {
        "train_size": len(train),
        "test_size": len(test),
        "accuracy": round(accuracy_score(test["label"], pred), 4),
        "fake_precision": round(float(p[0]), 4),
        "fake_recall": round(float(r[0]), 4),
        "fake_f1": round(float(f[0]), 4),
    }


def main() -> None:
    df = load()
    metrics = {
        "dataset": "Kaggle Fake and Real News (ISOT), deduplicated by text",
        "rows": len(df),
        "split": f"stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}, random_state={SEED}",
        "as_is": evaluate(df),
        "reuters_cues_removed": evaluate(df.assign(text=df["text"].str.replace(REUTERS_CUE, " ", regex=True))),
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    # Final model for the demo: same recipe, trained on all rows.
    joblib.dump(build().fit(df["text"], df["label"]), ROOT / "model" / "model.joblib", compress=3)


if __name__ == "__main__":
    main()
