from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).parent / "model.joblib"
FAKE_LABEL = 1


@lru_cache(maxsize=1)
def _load():
    return joblib.load(MODEL_PATH)


def predict(text: str) -> tuple[bool, float]:
    """Return (is_fake, probability of the predicted class)."""
    proba = _load().predict_proba([text])[0]
    is_fake = proba[FAKE_LABEL] >= 0.5
    return bool(is_fake), float(proba[FAKE_LABEL] if is_fake else 1 - proba[FAKE_LABEL])
