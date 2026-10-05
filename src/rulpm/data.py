"""Load NASA C-MAPSS turbofan data and build labels.

C-MAPSS files are space-separated text with 26 columns: unit id, cycle,
3 operational settings, 21 sensors. Train files run each engine to failure;
test files are truncated at a random cycle and RUL_FDxxx.txt gives the true
remaining useful life (RUL) at the last observed cycle of each test engine.
"""
from pathlib import Path

import pandas as pd

OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS

RUL_CAP = 125  # piecewise-linear RUL: early life is treated as "healthy"
FAIL_HORIZON = 30  # classification label: fails within N cycles


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, : len(COLUMNS)]  # some copies have trailing blanks
    df.columns = COLUMNS
    return df


def load_train(data_dir: str, subset: str) -> pd.DataFrame:
    return _read(Path(data_dir) / f"train_{subset}.txt")


def load_test(data_dir: str, subset: str):
    """Return (test_df, true_rul) where true_rul is indexed by unit id."""
    test = _read(Path(data_dir) / f"test_{subset}.txt")
    rul = pd.read_csv(Path(data_dir) / f"RUL_{subset}.txt", header=None)[0]
    rul.index = sorted(test["unit"].unique())
    return test, rul


def add_labels(df: pd.DataFrame, cap: int = RUL_CAP, horizon: int = FAIL_HORIZON) -> pd.DataFrame:
    """Add rul_raw, rul (capped) and fail_within_horizon for run-to-failure data."""
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul_raw"] = max_cycle - out["cycle"]
    out["rul"] = out["rul_raw"].clip(upper=cap)
    out["fail"] = (out["rul_raw"] <= horizon).astype(int)
    return out
