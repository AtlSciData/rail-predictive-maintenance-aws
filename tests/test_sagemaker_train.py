"""sagemaker/train.py end to end on synthetic data: uses real LightGBM if installed, else a stand-in."""
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "scripts", "sagemaker"):
    sys.path.insert(0, str(ROOT / p))

from make_synthetic import make  # noqa: E402
from rulpm.data import add_labels, load_train  # noqa: E402
from rulpm.export import export_normalized  # noqa: E402
from rulpm.features import feature_columns, rolling_features  # noqa: E402
import pandas as pd  # noqa: E402


def _install_fake_lightgbm():
    from sklearn.ensemble import RandomForestClassifier

    class LGBMClassifier:
        def __init__(self, **kw):
            self.m = RandomForestClassifier(n_estimators=20, random_state=kw.get("random_state", 0))

        def fit(self, X, y):
            self.m.fit(X, y)
            return self

        def predict_proba(self, X):
            return self.m.predict_proba(X)

        @property
        def booster_(self):
            return types.SimpleNamespace(save_model=lambda path: Path(path).write_text("fake"))

    sys.modules["lightgbm"] = types.SimpleNamespace(LGBMClassifier=LGBMClassifier)


def test_train_script_end_to_end(tmp_path, monkeypatch):
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        _install_fake_lightgbm()
    import train as sm_train

    make(tmp_path / "raw", n_train=12, seed=2)
    out = tmp_path / "norm"
    export_normalized(str(tmp_path / "raw"), "FD001", str(out))

    # what the Glue job would write: normalized CSV + rolling features (stand-in for Parquet)
    sensors = (out / "sensors.txt").read_text().split("\n")
    for split in ("train", "test"):
        df = pd.read_csv(out / split / "part.csv")
        feats = rolling_features(df, sensors, (5, 15))
        (tmp_path / "feat" / split).mkdir(parents=True)
        feats.to_csv(tmp_path / "feat" / split / "part.csv", index=False)
    (tmp_path / "labels").mkdir()
    (out / "test_rul.csv").replace(tmp_path / "labels" / "test_rul.csv")

    res = sm_train.run(str(tmp_path / "feat" / "train"), str(tmp_path / "feat" / "test"), str(tmp_path / "labels"),
                       str(tmp_path / "model"), folds=3, n_estimators=30)

    # same feature set as the local pipeline
    local_cols = feature_columns(pd.read_csv(tmp_path / "feat" / "train" / "part.csv"))
    assert json.loads((tmp_path / "model" / "feature_columns.json").read_text()) == local_cols
    for name in ("model.txt", "threshold.json", "metrics.json", "FD001_fail_lgbm_scores.json"):
        assert (tmp_path / "model" / name).exists()
    assert 0.0 <= res["test_tuned"]["threshold"] <= 1.0
    assert res["n_test_engines"] == 100 or res["n_test_engines"] > 0
    assert np.isfinite(res.get("test_roc_auc", 0.0))
