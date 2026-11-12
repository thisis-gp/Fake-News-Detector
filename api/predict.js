const { predict } = require("./_core");

const MAX_CHARS = 50000;

module.exports = (req, res) => {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const text = req.body && req.body.text;
  if (typeof text !== "string" || !text.trim()) return res.status(400).json({ error: "text is required" });
  res.status(200).json(predict(text.slice(0, MAX_CHARS)));
};
