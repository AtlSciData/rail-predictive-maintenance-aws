"""Build a sample request for the endpoint from the Glue test features.

Usage:
    python scripts/make_sample_payload.py --features data/glue_out/FD001/test \
        --model-dir data/sm_out --out data/sample_payload.json --engines 5
Takes the last observed cycle of the first N test engines (the same rows the model was evaluated on).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="data/glue_out/FD001/test")
ap.add_argument("--model-dir", default="data/sm_out")
ap.add_argument("--out", default="data/sample_payload.json")
ap.add_argument("--engines", type=int, default=5)
a = ap.parse_args()

cols = json.loads(Path(a.model_dir, "feature_columns.json").read_text())
files = sorted(f for f in Path(a.features).rglob("*.parquet") if not f.name.startswith(("_", ".")))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else \
    pd.concat([pd.read_csv(f) for f in Path(a.features).rglob("*.csv")], ignore_index=True)
last = df.sort_values(["unit", "cycle"]).groupby("unit").tail(1).sort_values("unit").head(a.engines)
payload = {"instances": [{c: float(r[c]) for c in cols} for _, r in last.iterrows()]}
Path(a.out).write_text(json.dumps(payload))
print(f"wrote {a.out}: {len(payload['instances'])} engines (units {list(last['unit'])}), {len(cols)} features each")
