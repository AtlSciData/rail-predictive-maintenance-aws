"""HTTP layer of inference/app.py with a stub model (LightGBM loading is exercised by the Docker run)."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flask")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import app as inf  # noqa: E402


class StubModel:
    def predict(self, X):
        return 1 / (1 + np.exp(-X[:, 0]))  # probability rises with the first feature


@pytest.fixture()
def client():
    inf.STATE.clear()
    inf.STATE.update({"model": StubModel(), "threshold": 0.6, "columns": ["a", "b"]})
    yield inf.app.test_client()
    inf.STATE.clear()


def test_ping(client):
    assert client.get("/ping").status_code == 200


def test_named_and_ordered_instances(client):
    r = client.post("/invocations", json={"instances": [{"a": 2.0, "b": 0.0}, [-2.0, 0.0]]})
    out = r.get_json()
    assert r.status_code == 200 and out["flag"] == [1, 0] and out["threshold"] == 0.6
    assert out["probability"][0] > 0.6 > out["probability"][1]


def test_csv_input(client):
    r = client.post("/invocations", data="2.0,0.0\n-2.0,0.0\n", content_type="text/csv")
    assert r.status_code == 200 and r.get_json()["flag"] == [1, 0]


def test_bad_requests(client):
    assert client.post("/invocations", json={"instances": [{"a": 1.0}]}).status_code == 400   # missing feature
    assert client.post("/invocations", json={"instances": [[1.0, 2.0, 3.0]]}).status_code == 400  # wrong width
    assert client.post("/invocations", json={"nope": 1}).status_code == 400
