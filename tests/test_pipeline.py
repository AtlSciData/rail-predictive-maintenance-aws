import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic import make  # noqa: E402
from rulpm.data import add_labels, load_test, load_train  # noqa: E402
from rulpm.features import RegimeNormalizer, informative_sensors, rolling_features  # noqa: E402
from rulpm.metrics import nasa_score  # noqa: E402
from rulpm.train_baseline import run  # noqa: E402


def test_labels_and_cap(tmp_path):
    make(tmp_path)
    df = add_labels(load_train(tmp_path, "FD001"))
    assert df.groupby("unit")["rul_raw"].min().eq(0).all()
    assert df["rul"].max() <= 125
    assert set(df["fail"].unique()) == {0, 1}


def test_constant_sensor_dropped(tmp_path):
    make(tmp_path)
    sensors = informative_sensors(load_train(tmp_path, "FD001"))
    assert "s1" not in sensors and len(sensors) == 20


def test_rolling_features_no_lookahead(tmp_path):
    make(tmp_path)
    df = load_train(tmp_path, "FD001")
    sensors = informative_sensors(df)[:3]
    a = rolling_features(df, sensors)
    # truncating each engine must not change earlier rows' features
    cut = df[df["cycle"] <= 50]
    b = rolling_features(cut, sensors)
    cols = [c for c in a.columns if "_mean" in c or "_slope" in c or "_std" in c]
    merged = a[a["cycle"] <= 50].reset_index(drop=True)[cols].to_numpy()
    assert np.allclose(merged, b[cols].to_numpy())


def test_regime_normalizer_stats(tmp_path):
    make(tmp_path, n_regimes=3, seed=1)
    df = load_train(tmp_path, "FD001")
    sensors = informative_sensors(df)
    out = RegimeNormalizer(3).fit(df, sensors).transform(df)
    assert abs(out[sensors[0]].mean()) < 0.5


def test_nasa_score_asymmetry():
    assert nasa_score([50], [60]) > nasa_score([50], [40])  # late is worse


def test_end_to_end_rf(tmp_path):
    make(tmp_path)
    r = run(str(tmp_path), "FD001", "rul", models=["rf"], folds=3, out_dir=str(tmp_path / "res"))
    assert r["rf"]["test_rmse"] < 60
    c = run(str(tmp_path), "FD001", "fail", models=["rf"], folds=3, out_dir=str(tmp_path / "res"))
    assert "recall" in c["rf"]
    assert 0 <= c["rf"]["tuned"]["threshold"] <= 1 and "missed_failures" in c["rf"]["tuned"]


def test_explore_runs(tmp_path):
    from rulpm.explore import explore

    make(tmp_path, n_regimes=3, seed=2)
    s = explore(str(tmp_path), "FD001", out_dir=str(tmp_path / "res"), tsne_n=300)
    assert s["n_sensors_kept"] == 20
    assert (tmp_path / "res" / "figs" / "FD001_pca.png").exists()
    assert (tmp_path / "res" / "figs" / "FD001_tsne.png").exists()


def test_export_normalized(tmp_path):
    import pandas as pd

    from rulpm.export import export_normalized

    make(tmp_path / "raw")
    sensors = export_normalized(str(tmp_path / "raw"), "FD001", str(tmp_path / "proc"))
    tr = pd.read_csv(tmp_path / "proc" / "train" / "part.csv")
    assert {"unit", "cycle", "regime", "rul", "fail"} <= set(tr.columns)
    assert len(sensors) == 20 and (tmp_path / "proc" / "test_rul.csv").exists()


def test_model_zoo_builds_distinct_classes():
    """Regression test: 'lgbm' and 'xgb' must build their own classes (late-binding bug).
    Uses fake lightgbm/xgboost modules so it runs without those packages."""
    import types

    from rulpm.models import get_models

    def fake(name, classes):
        mod = types.ModuleType(name)
        for c in classes:
            setattr(mod, c, type(c, (), {"__init__": lambda self, **kw: setattr(self, "kw", kw)}))
        return mod

    saved = {k: sys.modules.get(k) for k in ("lightgbm", "xgboost")}
    sys.modules["lightgbm"] = fake("lightgbm", ["LGBMRegressor", "LGBMClassifier"])
    sys.modules["xgboost"] = fake("xgboost", ["XGBRegressor", "XGBClassifier"])
    try:
        for task, (l_name, x_name) in {"rul": ("LGBMRegressor", "XGBRegressor"),
                                      "fail": ("LGBMClassifier", "XGBClassifier")}.items():
            zoo = get_models(task)
            lg, xg = zoo["lgbm"](), zoo["xgb"]()
            assert type(lg).__name__ == l_name and type(xg).__name__ == x_name
            assert "num_leaves" in lg.kw and "max_depth" in xg.kw
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_choose_threshold_meets_recall_on_validation():
    from rulpm.metrics import choose_threshold, threshold_report

    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(50)].astype(int)
    p = np.r_[rng.beta(2, 6, 200), rng.beta(5, 2, 50)]
    thr = choose_threshold(y, p, target_recall=0.9)
    rep = threshold_report(y, p, thr)
    assert rep["recall_at_thr"] >= 0.9
    # the highest qualifying threshold: raising it any further drops recall below target
    assert threshold_report(y, p, thr + 1e-9)["recall_at_thr"] < 0.9 or thr >= p.max()


def test_site_build_and_js_parity(tmp_path):
    import shutil
    import subprocess

    if shutil.which("node") is None:
        return  # node not installed: skip the JS parity check
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_demo_scores import make as make_scores
    from rulpm.build_site import build

    make_scores(tmp_path / "scores")
    build(str(tmp_path / "scores"), str(tmp_path / "web"))
    site = (tmp_path / "web" / "data" / "site-data.js")
    assert site.exists() and (tmp_path / "web" / "data" / "scores" / "FD001_fail_rf.json").exists()
    r = subprocess.run(["node", str(ROOT / "tests" / "js_parity.js"), str(site)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_count_regimes_single_vs_six():
    """Regression: a single noisy, quantized operating condition must not be split into clusters."""
    import pandas as pd

    from rulpm.explore import count_regimes

    rng = np.random.default_rng(0)
    n = 20000
    single = pd.DataFrame({"op1": rng.normal(0, 0.0022, n).round(4), "op2": rng.normal(0, 0.00029, n).round(4), "op3": 100.0})
    assert count_regimes(single)[0] == 1
    conds = np.array([[0, 0, 100], [10, 0.25, 100], [20, 0.7, 100], [25, 0.62, 60], [35, 0.84, 100], [42, 0.84, 100]], dtype=float)
    pick = conds[rng.integers(0, 6, n)] + rng.normal(0, [0.002, 0.0003, 0.0], size=(n, 3))
    six = pd.DataFrame(pick.round(4), columns=["op1", "op2", "op3"])
    assert count_regimes(six)[0] == 6
