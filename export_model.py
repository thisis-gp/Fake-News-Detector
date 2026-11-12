"""Export model/model.joblib to api/model.json for the JavaScript predictor."""
import json
from pathlib import Path

import joblib

ROOT = Path(__file__).parent
FEATURE_COUNT_ROUND = 8


def main() -> None:
    pipe = joblib.load(ROOT / "model" / "model.joblib")
    vec, clf = pipe.steps[0][1], pipe.steps[1][1]
    assert vec.ngram_range == (1, 1) and vec.lowercase and vec.norm == "l2" and vec.sublinear_tf and vec.smooth_idf
    assert vec.strip_accents is None and vec.token_pattern == r"(?u)\b\w\w+\b" and vec.use_idf
    words = [None] * len(vec.vocabulary_)
    for word, i in vec.vocabulary_.items():
        words[i] = word
    out = {
        "intercept": float(clf.intercept_[0]),
        "words": words,
        "idf": [round(float(x), FEATURE_COUNT_ROUND + 4) for x in vec.idf_],
        "coef": [round(float(x), FEATURE_COUNT_ROUND + 4) for x in clf.coef_[0]],
    }
    (ROOT / "api" / "model.json").write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
