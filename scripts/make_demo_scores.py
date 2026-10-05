"""Write NOISY synthetic score files (NOT real results) for testing the viewer.
Usage: python scripts/make_demo_scores.py <out_dir>   (never point this at results/scores)"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rulpm.metrics import choose_threshold, threshold_report  # noqa: E402


def make(out_dir, subset="FD001", n=100, seed=0):
    rng = np.random.default_rng(seed)
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    y = (rng.random(n) < 0.25).astype(int)
    for k, (name, sep) in enumerate({"rf": 2.4, "lgbm": 2.2, "xgb": 2.3}.items()):
        score = rng.normal(0, 1, n) + sep * y
        proba = 1 / (1 + np.exp(-(score - 1.2)))
        if name == "xgb":
            proba = np.round(proba, 2)  # force ties
        thr = choose_threshold(y, proba, 0.9)
        metrics = {"recall": 0, "roc_auc": float(roc_auc_score(y, proba)),
                   "pr_auc": float(average_precision_score(y, proba)), "target_recall": 0.9,
                   "tuned": threshold_report(y, proba, thr), "at_0.5": threshold_report(y, proba, 0.5)}
        (d / f"{subset}_fail_{name}.json").write_text(json.dumps({
            "subset": subset, "model": name, "task": "fail", "unit": list(range(1, n + 1)),
            "y_true": y.tolist(), "proba": [float(v) for v in proba], "python_metrics": metrics}))


if __name__ == "__main__":
    make(sys.argv[1])
