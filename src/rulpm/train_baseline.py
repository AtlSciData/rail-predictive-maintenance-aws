"""Baseline training / evaluation on one C-MAPSS subset.

Usage:
    python -m rulpm.train_baseline --subset FD001 --data-dir data/raw
Writes results/<subset>_<task>.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .data import FAIL_HORIZON, RUL_CAP, add_labels, load_test, load_train
from .features import RegimeNormalizer, feature_columns, informative_sensors, rolling_features
from .metrics import choose_threshold, classification_report, nasa_score, rmse, threshold_report
from .models import get_models

N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def prepare(data_dir: str, subset: str, windows=(5, 15)):
    train = add_labels(load_train(data_dir, subset))
    test, true_rul = load_test(data_dir, subset)
    sensors = informative_sensors(train)
    norm = RegimeNormalizer(N_REGIMES.get(subset, 1)).fit(train, sensors)
    tr = rolling_features(norm.transform(train), sensors, windows)
    te = rolling_features(norm.transform(test), sensors, windows)
    cols = feature_columns(tr)
    # evaluate on the last observed cycle of each test engine
    te_last = te.sort_values(["unit", "cycle"]).groupby("unit").tail(1).set_index("unit")
    te_last["rul_true"] = true_rul.clip(upper=RUL_CAP)
    te_last["fail_true"] = (true_rul <= FAIL_HORIZON).astype(int)
    return tr, te_last, cols


def save_scores(out_dir: str, subset: str, model: str, te_last, proba, metrics: dict):
    """Save test-engine probabilities and labels so curves can be recomputed and verified."""
    d = Path(out_dir, "scores")
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "subset": subset, "model": model, "task": "fail",
        "unit": [int(u) for u in te_last.index],
        "y_true": [int(v) for v in te_last["fail_true"]],
        "proba": [float(v) for v in proba],
        "python_metrics": metrics,
    }
    Path(d, f"{subset}_fail_{model}.json").write_text(json.dumps(payload))


def run(data_dir: str, subset: str, task: str, models=None, folds: int = 5, out_dir: str = "results",
        target_recall: float = 0.9):
    tr, te_last, cols = prepare(data_dir, subset)
    target = "rul" if task == "rul" else "fail"
    X, y, groups = tr[cols].to_numpy(), tr[target].to_numpy(), tr["unit"].to_numpy()
    results = {}
    for name, make in get_models(task, names=models).items():
        # engine-grouped CV: no engine appears in both train and validation folds
        cv = []
        oof = np.zeros(len(y))  # out-of-fold predictions, used to choose the threshold
        for tri, vai in GroupKFold(n_splits=folds).split(X, y, groups):
            m = make().fit(X[tri], y[tri])
            if task == "rul":
                cv.append(rmse(y[vai], m.predict(X[vai])))
            else:
                p = m.predict_proba(X[vai])[:, 1]
                oof[vai] = p
                cv.append(classification_report(y[vai], p).get("pr_auc", np.nan))
        m = make().fit(X, y)
        Xt = te_last[cols].to_numpy()
        if task == "rul":
            pred = m.predict(Xt)
            res = {"test_rmse": rmse(te_last["rul_true"], pred),
                   "test_nasa_score": nasa_score(te_last["rul_true"], pred)}
            res["cv_rmse_mean"] = float(np.mean(cv))
        else:
            proba = m.predict_proba(Xt)[:, 1]
            res = classification_report(te_last["fail_true"], proba)  # default 0.5 threshold
            res["cv_pr_auc_mean"] = float(np.nanmean(cv))
            thr = choose_threshold(y, oof, target_recall)  # chosen on validation data only
            res["target_recall"] = target_recall
            res["tuned"] = threshold_report(te_last["fail_true"], proba, thr)
            res["at_0.5"] = threshold_report(te_last["fail_true"], proba, 0.5)
            save_scores(out_dir, subset, name, te_last, proba, res)
        results[name] = res
        print(subset, task, name, json.dumps(res))
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir, f"{subset}_{task}.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=list(N_REGIMES))
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--task", default="rul", choices=["rul", "fail"])
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--target-recall", type=float, default=0.9)
    a = ap.parse_args()
    run(a.data_dir, a.subset, a.task, a.models, a.folds, target_recall=a.target_recall)
