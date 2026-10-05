"""Parity: PySpark rolling features == pandas rolling features. Skipped if pyspark is missing."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyspark")

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "scripts", "spark"):
    sys.path.insert(0, str(ROOT / p))

from make_synthetic import make  # noqa: E402
from glue_features import spark_rolling_features  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from rulpm.data import load_train  # noqa: E402
from rulpm.features import informative_sensors, rolling_features  # noqa: E402


def test_spark_matches_pandas(tmp_path):
    make(tmp_path, n_train=8, seed=3)
    df = load_train(tmp_path, "FD001")
    sensors = informative_sensors(df)[:4]
    keep = df[["unit", "cycle"] + sensors]
    expected = rolling_features(keep, sensors, (5, 15))
    spark = SparkSession.builder.master("local[1]").appName("parity").getOrCreate()
    got = spark_rolling_features(spark.createDataFrame(keep), sensors, (5, 15)).toPandas()
    got = got.sort_values(["unit", "cycle"]).reset_index(drop=True)
    assert list(got.columns) == list(expected.columns)
    assert np.allclose(got.drop(columns=["unit", "cycle"]).to_numpy(), expected.drop(columns=["unit", "cycle"]).to_numpy(), atol=1e-6)
    spark.stop()
