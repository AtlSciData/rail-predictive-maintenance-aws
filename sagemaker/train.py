"""SageMaker training entry point (script mode): LightGBM failure-within-30-cycles classifier.

Reads the Parquet features written by the Glue job, trains with engine-grouped CV, picks a
decision threshold on out-of-fold predictions (never on test), evaluates on each test engine's
last cycle, and writes the model + metrics to the model directory (SageMaker uploads it to S3).

Channels (SageMaker mounts S3 prefixes here):
    train   -> features/<subset>/train   (Parquet; includes unit, cycle, regime, sensors, rul, fail)
    test    -> features/<subset>/test    (Parquet; no labels)
    labels  -> normalized/<subset>/test_rul.csv  (true RUL at each test engine's last cycle)

Metric lines printed here are picked up by CloudWatch via the job's metric definitions.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:  # packaged next to this file by sagemaker/package.py
    from metrics import choose_threshold, classification_report, threshold_report
except ImportError:  # running from the repo
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rulpm"))
    from metrics import choose_threshold, classification_report, threshold_report

FAIL_HORIZON = 30
NOT_FEATURES = {"unit", "rul_raw", "rul", "fail", "op1", "op2", "op3"}  # same as rulpm.features.feature_columns


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    files = sorted(f for f in p.rglob("*.parquet") if not f.name.startswith(("_", ".")))
    if files:
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    csvs = sorted(p.rglob("*.csv"))  # fallback, used by the local test
    if not csvs:
        raise FileNotFoundError(f"no .parquet or .csv files under {path}")
    return pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)


def run(train_dir, test_dir, labels_dir, model_dir, subset="FD001", folds=5, target_recall=0.9,
        n_estimators=400, learning_rate=0.05, num_leaves=31, seed=0):
    import lightgbm as lgb

    tr = read_table(train_dir).sort_values(["unit", "cycle"]).reset_index(drop=True)
    te = read_table(test_dir).sort_values(["unit", "cycle"]).reset_index(drop=True)
    cols = [c for c in tr.columns if c not in NOT_FEATURES]
    missing = [c for c in cols if c not in te.columns]
    if missing:
        raise ValueError(f"test features missing columns: {missing[:5]}")
    X, y, groups = tr[cols].to_numpy(), tr["fail"].to_numpy(), tr["unit"].to_numpy()

    # evaluate on the last observed cycle of each test engine
    te_last = te.groupby("unit").tail(1).set_index("unit").copy()
    true_rul = read_table(labels_dir).set_index("unit")["rul_true"]
    te_last["fail_true"] = (true_rul.reindex(te_last.index) <= FAIL_HORIZON).astype(int)
    Xt, yt = te_last[cols].to_numpy(), te_last["fail_true"].to_numpy()

    def make():
        return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                                  subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1)

    oof = np.zeros(len(y))
    for tri, vai in GroupKFold(n_splits=folds).split(X, y, groups):  # no engine in both train and validation
        oof[vai] = make().fit(X[tri], y[tri]).predict_proba(X[vai])[:, 1]
    cv = classification_report(y, oof)
    thr = choose_threshold(y, oof, target_recall)  # chosen on validation data only

    model = make().fit(X, y)
    proba = model.predict_proba(Xt)[:, 1]
    test = classification_report(yt, proba)
    result = {
        "subset": subset, "model": "lgbm", "n_train_rows": int(len(tr)), "n_test_engines": int(len(te_last)),
        "n_features": len(cols), "cv_oof": cv, "test_at_0.5": threshold_report(yt, proba, 0.5),
        "test_tuned": threshold_report(yt, proba, thr), "target_recall": target_recall, **{f"test_{k}": v for k, v in test.items()},
    }

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out / "model.txt"))
    (out / "threshold.json").write_text(json.dumps({"threshold": thr, "target_recall": target_recall}))
    (out / "feature_columns.json").write_text(json.dumps(cols))
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / f"{subset}_fail_lgbm_scores.json").write_text(json.dumps({
        "subset": subset, "model": "lgbm", "task": "fail", "unit": [int(u) for u in te_last.index],
        "y_true": [int(v) for v in yt], "proba": [float(v) for v in proba]}))

    # CloudWatch metric lines (regexes live in the training job's metric definitions)
    print(f"test_roc_auc={result.get('test_roc_auc', float('nan')):.4f};")
    print(f"test_pr_auc={result.get('test_pr_auc', float('nan')):.4f};")
    print(f"test_recall_tuned={result['test_tuned']['recall_at_thr']:.4f};")
    print(f"test_precision_tuned={result['test_tuned']['precision_at_thr']:.4f};")
    print(f"cv_pr_auc={cv.get('pr_auc', float('nan')):.4f};")
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--target-recall", "--target_recall", dest="target_recall", type=float, default=0.9)
    ap.add_argument("--n-estimators", "--n_estimators", dest="n_estimators", type=int, default=400)
    ap.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", "--num_leaves", dest="num_leaves", type=int, default=31)
    a, _ = ap.parse_known_args()  # SageMaker adds a few of its own arguments
    run(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
        os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"),
        os.environ.get("SM_CHANNEL_LABELS", "/opt/ml/input/data/labels"),
        os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        a.subset, a.folds, a.target_recall, a.n_estimators, a.learning_rate, a.num_leaves)
