"""Build a 'drifted' copy of the sample request to trigger the CloudWatch drift alarm.

Usage:
    python scripts/make_drift_payload.py --payload data/sample_payload.json \
        --out data/drift_payload.json --shift 1.0
Adds --shift to every rolling-mean feature (columns containing "_mean"). The sensors are normalized,
so 1.0 is roughly one standard deviation. Try 0.5, 1, 2, or a negative value and look at the flag rate
the endpoint returns: the point is a visible change from the 0.23 baseline.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--payload", default="data/sample_payload.json")
ap.add_argument("--out", default="data/drift_payload.json")
ap.add_argument("--shift", type=float, default=1.0)
a = ap.parse_args()

p = json.loads(Path(a.payload).read_text())
n_cols = 0
for inst in p["instances"]:
    for k in inst:
        if "_mean" in k:
            inst[k] += a.shift
            n_cols += 1
Path(a.out).write_text(json.dumps(p))
print(f"wrote {a.out}: shifted {n_cols // max(len(p['instances']), 1)} mean features by {a.shift:+g} for {len(p['instances'])} engines")
