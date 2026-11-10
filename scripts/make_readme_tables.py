"""Build the README result tables from results/*.json and check the claims written under them.

Usage (from the project folder):
    python scripts/make_readme_tables.py
    python scripts/make_readme_tables.py --results-dir results --out build/readme_tables.md

Reads  results/FD00x_rul.json, results/FD00x_fail.json (written by rulpm.train_baseline, all three
models) and, if present, results/FD001_sagemaker_metrics.json.
Writes the two markdown tables to --out (UTF-8) and prints them with a list of facts to check
against the sentences in the README ("What the numbers show", the AWS paragraph).
"""
import argparse
import json
import sys
from pathlib import Path

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
MODELS = [("lgbm", "LightGBM"), ("xgb", "XGBoost"), ("rf", "Random Forest")]
ARROW = "→"


def load(results_dir, subset, task):
    path = Path(results_dir) / f"{subset}_{task}.json"
    if not path.exists():
        sys.exit(f"missing {path}: run rulpm.train_baseline for {subset} --task {task} (all models)")
    data = json.loads(path.read_text())
    absent = [m for m, _ in MODELS if m not in data]
    if absent:
        sys.exit(f"{path} lacks {absent}: it was written by a run with --models; rerun without --models")
    return data


def rul_table(results_dir):
    rows = ["| Subset | Model | Test RMSE | NASA score | CV RMSE |", "|---|---|---|---|---|"]
    for s in SUBSETS:
        r = load(results_dir, s, "rul")
        best = min(r[m]["test_rmse"] for m, _ in MODELS)
        for m, name in MODELS:
            v = r[m]
            rmse = f"{v['test_rmse']:.1f}"
            if v["test_rmse"] == best:
                rmse = f"**{rmse}**"
            rows.append(f"| {s} | {name} | {rmse} | {v['test_nasa_score']:.0f} | {v['cv_rmse_mean']:.1f} |")
    return "\n".join(rows)


def fail_table(results_dir):
    rows = ["| Subset | Model | ROC-AUC | PR-AUC | Recall @0.5 | Recall tuned | Precision tuned "
            f"| Missed (0.5 {ARROW} tuned) | False alarms (0.5 {ARROW} tuned) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s in SUBSETS:
        r = load(results_dir, s, "fail")
        for m, name in MODELS:
            v, a, t = r[m], r[m]["at_0.5"], r[m]["tuned"]
            rows.append(
                f"| {s} | {name} | {v['roc_auc']:.3f} | {v['pr_auc']:.3f} | {a['recall_at_thr']:.2f} "
                f"| {t['recall_at_thr']:.2f} | {t['precision_at_thr']:.2f} "
                f"| {a['missed_failures']} {ARROW} {t['missed_failures']} "
                f"| {a['false_alarms']} {ARROW} {t['false_alarms']} |")
    return "\n".join(rows)


def facts(results_dir):
    out = []
    rul = {s: load(results_dir, s, "rul") for s in SUBSETS}
    fail = {s: load(results_dir, s, "fail") for s in SUBSETS}
    rm = [rul[s][m]["test_rmse"] for s in SUBSETS for m, _ in MODELS]
    roc = [fail[s][m]["roc_auc"] for s in SUBSETS for m, _ in MODELS]
    out.append(f"RUL test RMSE range over all 12 pairs: {min(rm):.1f} to {max(rm):.1f}")
    out.append(f"ROC-AUC range over all 12 pairs: {min(roc):.3f} to {max(roc):.3f}")
    for s in SUBSETS:
        ranked = sorted(MODELS, key=lambda mn: rul[s][mn[0]]["test_rmse"])
        gap = rul[s][ranked[1][0]]["test_rmse"] - rul[s][ranked[0][0]]["test_rmse"]
        out.append(f"{s}: lowest RUL RMSE = {ranked[0][1]} (next best {ranked[1][1]}, gap {gap:.2f})")
    up = down = same = below = 0
    for s in SUBSETS:
        for m, _ in MODELS:
            v = fail[s][m]
            r5, rt = v["at_0.5"]["recall_at_thr"], v["tuned"]["recall_at_thr"]
            up += rt > r5
            down += rt < r5
            same += rt == r5
            below += rt < v["target_recall"]
    out.append(f"Tuned threshold vs 0.5, recall raised/lowered/unchanged in {up}/{down}/{same} of 12 pairs")
    out.append(f"Tuned recall below the 0.9 target on the test engines in {below} of 12 pairs")
    sm_path = Path(results_dir) / "FD001_sagemaker_metrics.json"
    if sm_path.exists():
        sm = json.loads(sm_path.read_text())
        loc = fail["FD001"]["lgbm"]
        t = loc["tuned"]
        st = sm["test_tuned"]
        out.append("FD001 LightGBM local vs SageMaker (AWS paragraph in README):")
        out.append(f"  local     ROC {loc['roc_auc']:.3f} PR {loc['pr_auc']:.3f} thr {t['threshold']:.4f} "
                   f"recall {t['recall_at_thr']:.2f} precision {t['precision_at_thr']:.2f} "
                   f"flagged {t['flagged']} missed {t['missed_failures']} false alarms {t['false_alarms']}")
        out.append(f"  sagemaker ROC {sm['test_roc_auc']:.3f} PR {sm['test_pr_auc']:.3f} thr {st['threshold']:.4f} "
                   f"recall {st['recall_at_thr']:.2f} precision {st['precision_at_thr']:.2f} "
                   f"flagged {st['flagged']} missed {st['missed_failures']} false alarms {st['false_alarms']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="build/readme_tables.md")
    a = ap.parse_args()
    text = ("**Remaining useful life (RUL, capped at 125)** - best test RMSE per subset in bold\n\n"
            + rul_table(a.results_dir)
            + "\n\n**Failure within 30 cycles.**\n\n" + fail_table(a.results_dir) + "\n")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    print("Facts to check against the README sentences:")
    for line in facts(a.results_dir):
        print(" -", line)
    print(f"\nTables written to {out} (UTF-8). Open it and copy the two tables into README.md.")


if __name__ == "__main__":
    main()
