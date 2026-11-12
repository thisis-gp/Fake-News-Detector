// Same maths as sklearn TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2) + LogisticRegression.
// Stop words are not in the vocabulary, so unknown tokens (including stop words) are simply skipped.
const model = require("./model.json");

const index = new Map(model.words.map((w, i) => [w, i]));
const TOKEN = /\p{L}[\p{L}\p{N}_]+|\p{N}[\p{L}\p{N}_]+|_[\p{L}\p{N}_]+/gu; // (?u)\b\w\w+\b
const TOP_WORDS = 6;

function predict(text) {
  const counts = new Map();
  for (const tok of text.toLowerCase().match(TOKEN) || []) {
    const i = index.get(tok);
    if (i !== undefined) counts.set(i, (counts.get(i) || 0) + 1);
  }
  let sumSq = 0;
  const weights = [];
  for (const [i, c] of counts) {
    const w = (1 + Math.log(c)) * model.idf[i];
    weights.push([i, w]);
    sumSq += w * w;
  }
  const norm = Math.sqrt(sumSq) || 1;
  let z = model.intercept;
  const parts = [];
  for (const [i, w] of weights) {
    const contribution = (model.coef[i] * w) / norm;
    z += contribution;
    parts.push([model.words[i], contribution]);
  }
  const pFake = 1 / (1 + Math.exp(-z));
  const isFake = pFake >= 0.5;
  const sign = isFake ? 1 : -1;
  const top = parts
    .filter(([, c]) => c * sign > 0)
    .sort((a, b) => b[1] * sign - a[1] * sign)
    .slice(0, TOP_WORDS)
    .map(([word, c]) => ({ word, weight: Math.abs(c) }));
  return { label: isFake ? "fake" : "true", pFake, confidence: isFake ? pFake : 1 - pFake, knownWords: counts.size, topWords: top };
}

module.exports = { predict };
