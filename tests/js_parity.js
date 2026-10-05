// Usage: node tests/js_parity.js web/data/site-data.js  -> exits 1 on any mismatch with Python's AUCs
const fs = require("fs"), path = require("path");
const M = require(path.join(__dirname, "..", "web", "metrics.js"));
global.window = {};
eval(fs.readFileSync(process.argv[2], "utf8"));
let bad = 0, n = 0;
for (const [s, models] of Object.entries(window.SITE_DATA.subsets))
  for (const [m, d] of Object.entries(models)) {
    const cv = M.curves(d.y_true, d.proba);
    const a = M.rocAuc(cv.roc), b = M.averagePrecision(cv.pr);
    const ok = Math.abs(a - d.python.roc_auc) < 1e-9 && Math.abs(b - d.python.pr_auc) < 1e-9;
    n++; if (!ok) { bad++; console.log("MISMATCH", s, m, a, d.python.roc_auc, b, d.python.pr_auc); }
  }
console.log(`${n} model(s) checked, ${bad} mismatch(es)`);
process.exit(bad ? 1 : 0);
