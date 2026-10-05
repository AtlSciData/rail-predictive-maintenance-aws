// Curve math shared by the page and the node test. Matches scikit-learn:
//   roc_auc_score (trapezoid over distinct thresholds) and average_precision_score (step sum).
(function (root) {
  function curves(y, p) {
    const n = y.length;
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[b] - p[a]);
    let P = 0; for (const v of y) P += v;
    const N = n - P;
    const roc = [{ thr: Infinity, fpr: 0, tpr: 0 }];
    const pr = [];
    let tp = 0, fp = 0, i = 0;
    while (i < n) {
      const s = p[idx[i]];
      while (i < n && p[idx[i]] === s) { if (y[idx[i]] === 1) tp++; else fp++; i++; }
      roc.push({ thr: s, fpr: N ? fp / N : 0, tpr: P ? tp / P : 0 });
      pr.push({ thr: s, recall: P ? tp / P : 0, precision: tp / (tp + fp) });
    }
    return { roc, pr, P, N };
  }
  function rocAuc(roc) {
    let a = 0;
    for (let i = 1; i < roc.length; i++) a += (roc[i].fpr - roc[i - 1].fpr) * (roc[i].tpr + roc[i - 1].tpr) / 2;
    return a;
  }
  function averagePrecision(pr) {
    let ap = 0, prev = 0;
    for (const q of pr) { ap += (q.recall - prev) * q.precision; prev = q.recall; }
    return ap;
  }
  function atThreshold(y, p, t) {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (let i = 0; i < y.length; i++) {
      const pos = p[i] >= t;
      if (pos && y[i] === 1) tp++; else if (pos) fp++; else if (y[i] === 1) fn++; else tn++;
    }
    return { tp, fp, fn, tn, flagged: tp + fp,
             recall: tp + fn ? tp / (tp + fn) : 0, precision: tp + fp ? tp / (tp + fp) : null };
  }
  const api = { curves, rocAuc, averagePrecision, atThreshold };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.CurveMath = api;
})(typeof window !== "undefined" ? window : globalThis);
