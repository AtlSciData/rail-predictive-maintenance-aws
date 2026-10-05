"""SHAP explainability for a tree model trained on one subset.

Usage:
    python -m rulpm.explain --subset FD001 --model lgbm
Writes results/figs/<subset>_shap_summary.png and results/<subset>_shap_top.json
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .models import get_models  # noqa: E402
from .train_baseline import N_REGIMES, prepare  # noqa: E402


def explain(data_dir: str, subset: str, model: str = "lgbm", out_dir: str = "results", n: int = 2000, seed: int = 0):
    import shap  # imported here so the rest of the package works without it

    tr, te_last, cols = prepare(data_dir, subset)
    make = get_models("rul", names=[model])[model]
    m = make().fit(tr[cols].to_numpy(), tr["rul"].to_numpy())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tr), size=min(n, len(tr)), replace=False)
    Xs = tr[cols].iloc[idx]
    sv = shap.TreeExplainer(m).shap_values(Xs)
    importance = np.abs(sv).mean(axis=0)
    top = sorted(zip(cols, importance), key=lambda t: -t[1])[:15]
    Path(out_dir, "figs").mkdir(parents=True, exist_ok=True)
    shap.summary_plot(sv, Xs, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(Path(out_dir, "figs", f"{subset}_shap_summary.png"), dpi=130)
    plt.close()
    Path(out_dir, f"{subset}_shap_top.json").write_text(json.dumps([[c, float(v)] for c, v in top], indent=2))
    for c, v in top:
        print(f"{c:20s} {v:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=list(N_REGIMES))
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--model", default="lgbm", choices=["rf", "lgbm", "xgb"])
    a = ap.parse_args()
    explain(a.data_dir, a.subset, a.model)
