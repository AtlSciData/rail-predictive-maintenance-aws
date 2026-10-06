"""SageMaker-compatible inference server for the LightGBM failure classifier.

SageMaker contract: GET /ping -> 200 when ready, POST /invocations -> predictions, port 8080.
The trained model is unpacked by SageMaker into /opt/ml/model (model.txt, threshold.json,
feature_columns.json, written by sagemaker/train.py).

Request  (application/json):
    {"instances": [ {"s2_mean5": 0.1, ...}, ... ]}      features by name   (preferred)
    {"instances": [ [0.1, 0.2, ...], ... ]}             features in feature_columns.json order
    (text/csv is also accepted: one row per line, values in feature_columns.json order)
Response (application/json):
    {"probability": [...], "flag": [0|1, ...], "threshold": 0.11, "n_features": 107}
flag = 1 when probability >= the threshold chosen on out-of-fold validation data during training.

Every request also logs one JSON line ("event": "prediction_batch") with batch size, mean
probability, flag rate and latency, so CloudWatch can chart or alarm on prediction drift.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/ml/model"))
app = Flask(__name__)
STATE = {}


def load_model(model_dir=MODEL_DIR):
    import lightgbm as lgb  # imported here so the HTTP layer can be tested without LightGBM

    model_dir = Path(model_dir)
    return {
        "model": lgb.Booster(model_file=str(model_dir / "model.txt")),
        "threshold": float(json.loads((model_dir / "threshold.json").read_text())["threshold"]),
        "columns": json.loads((model_dir / "feature_columns.json").read_text()),
    }


def state():
    if not STATE:
        STATE.update(load_model())
    return STATE


class BadRequest(Exception):
    pass


def parse_features(columns):
    ctype = (request.content_type or "").split(";")[0].strip().lower()
    if ctype == "text/csv":
        rows = [r for r in request.get_data(as_text=True).splitlines() if r.strip()]
        data = [[float(v) if v.strip() else np.nan for v in r.split(",")] for r in rows]
    else:
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("instances"), list) or not body["instances"]:
            raise BadRequest('expected JSON like {"instances": [ {...}, ... ]}')
        data = []
        for inst in body["instances"]:
            if isinstance(inst, dict):
                missing = [c for c in columns if c not in inst]
                if missing:
                    raise BadRequest(f"missing {len(missing)} features, e.g. {missing[:3]}")
                data.append([np.nan if inst[c] is None else float(inst[c]) for c in columns])
            elif isinstance(inst, list):
                data.append([np.nan if v is None else float(v) for v in inst])
            else:
                raise BadRequest("each instance must be an object or a list")
    X = np.asarray(data, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(columns):
        raise BadRequest(f"expected {len(columns)} features per row, got shape {list(X.shape)}")
    return X


@app.get("/ping")
def ping():
    try:
        state()
        return Response("\n", status=200)
    except Exception as e:  # model not loadable -> unhealthy
        print(json.dumps({"event": "load_error", "error": str(e)}), flush=True)
        return Response("\n", status=500)


@app.post("/invocations")
def invocations():
    t0 = time.time()
    try:
        s = state()
        X = parse_features(s["columns"])
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    proba = np.asarray(s["model"].predict(X), dtype=float)
    flag = (proba >= s["threshold"]).astype(int)
    print(json.dumps({
        "event": "prediction_batch", "n": int(len(proba)), "mean_proba": float(proba.mean()),
        "flag_rate": float(flag.mean()), "latency_ms": round((time.time() - t0) * 1000, 2),
    }), flush=True)
    return jsonify({"probability": proba.tolist(), "flag": flag.tolist(),
                    "threshold": s["threshold"], "n_features": len(s["columns"])})
