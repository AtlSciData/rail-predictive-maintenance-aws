"""Rolling-window features in PySpark - the same definitions as rulpm.features.rolling_features.

Runs as an AWS Glue job (or locally with pyspark) on normalized CSV data:
    --input_s3   s3://bucket/normalized/FD001/train    (CSV with header)
    --output_s3  s3://bucket/features/FD001/train      (written as Parquet)
    --windows    5,15
    --sensors_file  optional; default = every column starting with 's'

Feature definitions (match the pandas version):
    mean_w, std_w  over rows [t-w+1, t] of the engine (partial windows at the start; std = 0 if 1 row)
    slope_w        = (x_t - x_{t-w+1}) / (w-1), 0 until a full window exists
"""
import argparse
import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def spark_rolling_features(df, sensors, windows=(5, 15)):
    order = Window.partitionBy("unit").orderBy("cycle")
    cols = list(df.columns)
    out = df
    for w in windows:
        win = order.rowsBetween(-(w - 1), 0)
        for s in sensors:
            out = (
                out.withColumn(f"{s}_mean{w}", F.avg(s).over(win))
                .withColumn(f"{s}_std{w}", F.coalesce(F.stddev_samp(s).over(win), F.lit(0.0)))
                .withColumn(
                    f"{s}_slope{w}",
                    F.coalesce((F.col(s) - F.lag(s, w - 1).over(order)) / F.lit(float(w - 1)), F.lit(0.0)),
                )
            )
    return out.select(cols + [c for c in out.columns if c not in cols])


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_s3", required=True)
    ap.add_argument("--output_s3", required=True)
    ap.add_argument("--windows", default="5,15")
    ap.add_argument("--sensors", default="")
    a, _ = ap.parse_known_args(argv)  # Glue passes extra args such as --JOB_NAME
    return a


def main(argv=None):
    a = parse_args(argv or sys.argv[1:])
    spark = SparkSession.builder.appName("rulpm-features").getOrCreate()
    df = spark.read.csv(a.input_s3, header=True, inferSchema=True)
    sensors = a.sensors.split(",") if a.sensors else [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
    feats = spark_rolling_features(df, sensors, tuple(int(w) for w in a.windows.split(",")))
    feats.write.mode("overwrite").parquet(a.output_s3)
    spark.stop()


if __name__ == "__main__":
    main()
