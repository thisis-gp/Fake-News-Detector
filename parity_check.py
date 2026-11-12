"""Compare sklearn and the JS predictor on held-out articles. Usage: python parity_check.py"""
import json, subprocess, sys
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from train import SEED, TEST_SIZE, load

ROOT = Path(__file__).parent
N = 2000

df = load()
_, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"])
texts = test["text"].head(N).tolist()
texts += ["", "the and of", "Ünïcode café naïve 2020 x_y __a 12ab", "BREAKING!!! You won't BELIEVE this"]
skl = joblib.load(ROOT / "model" / "model.joblib").predict_proba(texts)[:, 1]
js = subprocess.run(["node", str(ROOT / "parity_check.js")], input=json.dumps(texts), capture_output=True, text=True, check=True).stdout
js = json.loads(js)
diffs = [abs(a - b) for a, b in zip(skl, js)]
print(f"articles={len(texts)} max_abs_diff={max(diffs):.3e}")
sys.exit(0 if max(diffs) < 1e-6 else 1)
