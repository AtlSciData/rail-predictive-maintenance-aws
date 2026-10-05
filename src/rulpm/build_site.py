"""Assemble the static viewer data from saved test-engine scores.

Usage:
    python -m rulpm.build_site            # reads results/scores/*.json, writes web/data/
Then open the page:
    python -m http.server 8000 --directory web      ->  http://localhost:8000
(web/index.html also works when opened directly, because the data is loaded as a script.)

Only derived data (model scores, labels, metrics) is published - not the NASA files.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def build(scores_dir: str = "results/scores", web_dir: str = "web") -> dict:
    src = Path(scores_dir)
    out = Path(web_dir, "data")
    (out / "scores").mkdir(parents=True, exist_ok=True)
    subsets: dict = {}
    for f in sorted(src.glob("*_fail_*.json")):
        payload = json.loads(f.read_text())
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        shutil.copyfile(f, out / "scores" / f.name)
        pm = payload["python_metrics"]
        tuned = pm.get("tuned", {})
        subsets.setdefault(payload["subset"], {})[payload["model"]] = {
            "file": f"data/scores/{f.name}",
            "sha256": sha,
            "n": len(payload["y_true"]),
            "positives": int(sum(payload["y_true"])),
            "y_true": payload["y_true"],
            "proba": payload["proba"],
            "python": {
                "roc_auc": pm.get("roc_auc"),
                "pr_auc": pm.get("pr_auc"),
                "tuned_threshold": tuned.get("threshold"),
                "target_recall": pm.get("target_recall"),
            },
        }
    site = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "subsets": subsets}
    Path(out, "site-data.js").write_text("window.SITE_DATA = " + json.dumps(site) + ";\n")
    return site


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default="results/scores")
    ap.add_argument("--web-dir", default="web")
    a = ap.parse_args()
    s = build(a.scores_dir, a.web_dir)
    print("subsets:", {k: sorted(v) for k, v in s["subsets"].items()})
