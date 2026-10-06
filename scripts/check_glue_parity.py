"""Compare Glue/PySpark features (Parquet downloaded from S3) with the pandas implementation.

Usage (from the project folder):
    python scripts/check_glue_parity.py --csv data/processed/FD001/train/part.csv \
        --parquet data/glue_out/FD001/train

--csv      the normalized CSV that was uploaded to S3 as the Glue job's input
--parquet  folder holding the .parquet files the Glue job wrote (downloaded from S3)

Needs pandas + pyarrow (pip install pyarrow). Exit code 0 = match, 1 = mismatch.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rulpm.features import rolling_features  # noqa: E402

KEYS = ["unit", "cycle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--windows", default="5,15")
    ap.add_argument("--atol", type=float, default=1e-6)
    a = ap.parse_args()
    windows = tuple(int(w) for w in a.windows.split(","))

    src = pd.read_csv(a.csv)
    # Same default as the Glue job: columns named s<number>
    sensors = [c for c in src.columns if c.startswith("s") and c[1:].isdigit()]
    expected = rolling_features(src, sensors, windows)

    files = sorted(Path(a.parquet).rglob("*.parquet"))
    if not files:
        sys.exit(f"No .parquet files found under {a.parquet}")
    got = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    got = got.sort_values(KEYS).reset_index(drop=True)
    expected = expected.sort_values(KEYS).reset_index(drop=True)

    feat_cols = [c for c in expected.columns if any(t in c for t in ("_mean", "_std", "_slope"))]

    print(f"rows: csv={len(src)}  parquet={len(got)}  sensors={len(sensors)}  feature columns={len(feat_cols)}")
    problems = []
    if len(got) != len(expected):
        problems.append("row counts differ")
    missing = [c for c in feat_cols if c not in got.columns]
    if missing:
        problems.append(f"{len(missing)} feature columns missing in Parquet, e.g. {missing[:3]}")
    if not (got[KEYS].to_numpy() == expected[KEYS].to_numpy()).all():
        problems.append("(unit, cycle) keys do not line up")

    if not problems:
        diff = (got[feat_cols].to_numpy(dtype=float) - expected[feat_cols].to_numpy(dtype=float))
        worst = float(np.nanmax(np.abs(diff)))
        bad_cols = [c for c, d in zip(feat_cols, np.nanmax(np.abs(diff), axis=0)) if d > a.atol]
        print(f"max absolute difference: {worst:.3e}  (tolerance {a.atol:g})")
        if bad_cols:
            problems.append(f"{len(bad_cols)} columns exceed tolerance, e.g. {bad_cols[:3]}")

    if problems:
        print("MISMATCH:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("MATCH: Spark features equal the pandas features.")


if __name__ == "__main__":
    main()
