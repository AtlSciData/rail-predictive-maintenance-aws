"""Leakage-safe feature engineering for C-MAPSS.

Steps (all fit on training engines only):
  1. drop constant / near-constant sensors
  2. cluster operating settings into regimes (KMeans) and z-score each sensor
     within its regime (FD002 / FD004 have six operating conditions)
  3. per-engine rolling mean / std / slope features

The same rolling definitions are re-implemented in PySpark (spark/glue_features.py)
so a parity test can check the two pipelines agree.
"""
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .data import OP_COLS, SENSOR_COLS


def informative_sensors(train: pd.DataFrame, min_std: float = 1e-6) -> List[str]:
    stds = train[SENSOR_COLS].std()
    return [c for c in SENSOR_COLS if stds[c] > min_std]


class RegimeNormalizer:
    """Cluster operating settings, then standardize sensors within each regime."""

    def __init__(self, n_regimes: int = 1, random_state: int = 0):
        self.n_regimes = n_regimes
        self.random_state = random_state

    def fit(self, train: pd.DataFrame, sensors: Sequence[str]):
        self.sensors = list(sensors)
        if self.n_regimes > 1:
            self.km_ = KMeans(self.n_regimes, n_init=10, random_state=self.random_state)
            self.km_.fit(train[OP_COLS])
        reg = self._regime(train)
        grp = train[self.sensors].groupby(reg)
        self.mean_ = grp.mean()
        self.std_ = grp.std().replace(0, 1.0).fillna(1.0)
        return self

    def _regime(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_regimes == 1:
            return np.zeros(len(df), dtype=int)
        return self.km_.predict(df[OP_COLS])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        reg = self._regime(df)
        out["regime"] = reg
        mean = self.mean_.loc[reg].to_numpy()
        std = self.std_.loc[reg].to_numpy()
        out[self.sensors] = (df[self.sensors].to_numpy() - mean) / std
        return out


def rolling_features(df: pd.DataFrame, sensors: Sequence[str], windows=(5, 15)) -> pd.DataFrame:
    """Per-engine rolling mean, std and slope for each sensor.

    slope = (x_t - x_{t-w+1}) / (w - 1), filled with 0 until a full window exists.
    Features use only current and past cycles (no look-ahead).
    """
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit")
    feats = {}
    for w in windows:
        for s in sensors:
            roll = g[s].rolling(w, min_periods=1)
            feats[f"{s}_mean{w}"] = roll.mean().reset_index(level=0, drop=True)
            feats[f"{s}_std{w}"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
            feats[f"{s}_slope{w}"] = ((df[s] - g[s].shift(w - 1)) / (w - 1)).fillna(0.0)
    return pd.concat([df, pd.DataFrame(feats)], axis=1)


def feature_columns(df: pd.DataFrame) -> List[str]:
    drop = {"unit", "rul_raw", "rul", "fail"}
    return [c for c in df.columns if c not in drop and c not in OP_COLS]
