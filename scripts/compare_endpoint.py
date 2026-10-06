"""Check that endpoint predictions equal the probabilities saved by the training job.

Usage:
    python scripts/compare_endpoint.py --payload data/sample_payload.json \
        --response data/endpoint_out.json --scores data/sm_out/FD001_fail_lgbm_scores.json
Exit code 0 when the largest difference is below --tol.
"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--payload", default="data/sample_payload.json")
ap.add_argument("--response", required=True)
ap.add_argument("--scores", default="data/sm_out/FD001_fail_lgbm_scores.json")
ap.add_argument("--tol", type=float, default=1e-6)
a = ap.parse_args()

payload = json.loads(Path(a.payload).read_text()); units = payload.get("units")
got = json.loads(Path(a.response).read_text())["probability"]
sc = json.loads(Path(a.scores).read_text())
ref = dict(zip(sc["unit"], sc["proba"]))
units = units or sorted(ref)[:len(got)]
if len(units) != len(got):
    sys.exit(f"payload has {len(units)} engines but the response has {len(got)} predictions")
diffs = [abs(p - ref[u]) for u, p in zip(units, got)]
worst = max(diffs)
print(f"engines compared: {len(diffs)}   largest difference: {worst:.3e}   (tolerance {a.tol:g})")
print("MATCH" if worst <= a.tol else "MISMATCH")
sys.exit(0 if worst <= a.tol else 1)
