"""Step A of the AWS pipeline (pandas): fit the normalizer on training engines and
write normalized train/test data as CSV, ready to upload to S3 for the Glue job.

Usage:
    python -m rulpm.export --subset FD001 --data-dir data/raw --out-dir data/processed/FD001
Then:  aws s3 sync data/processed/FD001 s3://<bucket>/normalized/FD001
"""
import argparse
from pathlib import Path

from .data import add_labels, load_test, load_train
from .features import RegimeNormalizer, informative_sensors
from .train_baseline import N_REGIMES


def export_normalized(data_dir: str, subset: str, out_dir: str):
    train = add_labels(load_train(data_dir, subset))
    test, true_rul = load_test(data_dir, subset)
    sensors = informative_sensors(train)
    norm = RegimeNormalizer(N_REGIMES.get(subset, 1)).fit(train, sensors)
    out = Path(out_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    keep = ["unit", "cycle", "regime"] + sensors
    norm.transform(train)[keep + ["rul", "rul_raw", "fail"]].to_csv(out / "train" / "part.csv", index=False)
    norm.transform(test)[keep].to_csv(out / "test" / "part.csv", index=False)
    true_rul.rename("rul_true").rename_axis("unit").to_csv(out / "test_rul.csv")
    (out / "sensors.txt").write_text("\n".join(sensors))
    return sensors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=list(N_REGIMES))
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    export_normalized(a.data_dir, a.subset, a.out_dir or f"data/processed/{a.subset}")
