const { predict } = require("./api/_core");
let s = "";
process.stdin.on("data", (d) => (s += d)).on("end", () => {
  console.log(JSON.stringify(JSON.parse(s).map((t) => predict(t).pFake)));
});
