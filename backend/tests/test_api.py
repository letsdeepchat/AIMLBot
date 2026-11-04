"""HTTP API tests. Skipped automatically when FastAPI isn't installed."""
from __future__ import annotations

import importlib
import io
import os
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from tests.test_core import _binary_data  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    os.environ["QUEST_DATA_DIR"] = str(tmp_path)
    os.environ["QUEST_JOB_BACKEND"] = "inline"
    os.environ.pop("QUEST_DB_PATH", None)
    os.environ.pop("QUEST_STATIC_DIR", None)
    import app.main as main

    importlib.reload(main)  # pick up the temp data dir
    with TestClient(main.app) as c:
        yield c


def _files(n=300, n_test=120):
    train, test = _binary_data(n=n, n_test=n_test)
    return {
        "train": ("train.csv", io.BytesIO(train.to_csv(index=False).encode()), "text/csv"),
        "test": ("test.csv", io.BytesIO(test.to_csv(index=False).encode()), "text/csv"),
    }


def _wait_idle(client, challenge_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/challenges/{challenge_id}").json()
        if body["active_job"] is None:
            return body
        time.sleep(0.3)
    raise TimeoutError


def test_create_analyse_baseline_download(client):
    r = client.post("/api/challenges", data={"name": "Churn demo", "target": "target", "id_col": "row_id"},
                    files=_files())
    assert r.status_code == 201, r.text
    challenge = r.json()
    assert challenge["task"] == "binary" and challenge["spec"]["metric"] == "roc_auc"

    body = _wait_idle(client, challenge["id"])  # analysis starts automatically
    assert body["reports"]["analysis"] is True
    analysis = client.get(f"/api/challenges/{challenge['id']}/reports/analysis").json()
    assert analysis["cv_plan"]["strategy"] == "repeated_stratified_kfold"

    assert client.post(f"/api/challenges/{challenge['id']}/jobs", json={"kind": "baselines"}).status_code == 202
    body = _wait_idle(client, challenge["id"])
    assert body["reports"]["baselines"] is True
    baselines = client.get(f"/api/challenges/{challenge['id']}/reports/baselines").json()
    best = baselines["models"][0]["key"]
    csv = client.get(f"/api/challenges/{challenge['id']}/submissions/{best}")
    assert csv.status_code == 200 and csv.text.startswith("row_id,target")
    assert "churn-demo" in csv.headers["content-disposition"]

    assert client.delete(f"/api/challenges/{challenge['id']}").status_code == 204
    assert client.get(f"/api/challenges/{challenge['id']}").status_code == 404


def test_rejects_bad_inputs(client):
    r = client.post("/api/challenges", data={"name": "x", "target": "nope"}, files=_files())
    assert r.status_code == 400 and "nope" in r.json()["detail"]
    r = client.post("/api/challenges", data={"name": "x", "target": "target", "metric": "rmse"}, files=_files())
    assert r.status_code == 400 and "binary" in r.json()["detail"]
    files = _files()
    files["train"] = ("train.xlsx", io.BytesIO(b"x"), "application/octet-stream")
    assert client.post("/api/challenges", data={"name": "x", "target": "target"}, files=files).status_code == 400
    assert client.get("/api/challenges").json() == []  # failed uploads leave nothing behind
    assert client.get("/api/challenges/missing/reports/analysis").status_code == 404


def _ratio_files():
    from tests.test_experiments import _ratio_data

    train, test = _ratio_data(n=300, n_test=100)
    return {
        "train": ("train.csv", io.BytesIO(train.to_csv(index=False).encode()), "text/csv"),
        "test": ("test.csv", io.BytesIO(test.to_csv(index=False).encode()), "text/csv"),
    }


def test_experiments_settings_and_leaderboard(client):
    cid = client.post("/api/challenges", data={"name": "Ratio", "target": "target", "id_col": "row_id"},
                      files=_ratio_files()).json()["id"]
    _wait_idle(client, cid)
    assert client.post(f"/api/challenges/{cid}/experiments", json={"name": "x", "model": "linear", "steps": []}).status_code == 409
    assert client.post(f"/api/challenges/{cid}/jobs", json={"kind": "baselines"}).status_code == 202
    _wait_idle(client, cid)

    catalog = client.get("/api/recipe-steps").json()
    assert {s["type"] for s in catalog} >= {"formula", "target_encode", "group_agg"}

    steps = [{"type": "formula", "params": {"name": "closeness", "expression": "abs(log(a / b))"}}]
    check = client.post(f"/api/challenges/{cid}/recipes/validate", json={"steps": steps}).json()
    assert check["errors"] == [] and "closeness" in {c["name"] for c in check["output"]}
    bad = client.post(f"/api/challenges/{cid}/experiments",
                      json={"name": "bad", "model": "linear", "steps": [{"type": "log1p", "params": {"columns": ["shop"]}}]})
    assert bad.status_code == 400 and "Step 1" in bad.json()["detail"]

    created = client.post(f"/api/challenges/{cid}/experiments",
                          json={"name": "Closeness", "model": "linear", "steps": steps, "hypothesis": "a near b"})
    assert created.status_code == 201, created.text
    exp_id = created.json()["id"]
    _wait_idle(client, cid)
    listed = client.get(f"/api/challenges/{cid}/experiments").json()
    assert listed[0]["result"]["vs_control"]["verdict"] == "better" and listed[0]["stale"] is False
    assert "fold_scores" not in listed[0]["result"]
    board = client.get(f"/api/challenges/{cid}/leaderboard").json()
    assert board["entries"][0]["id"] == f"experiment:{exp_id}"
    sub = client.get(f"/api/challenges/{cid}/experiments/{exp_id}/submission")
    assert sub.status_code == 200 and sub.text.startswith("row_id,target")

    cols = client.get(f"/api/challenges/{cid}/columns").json()
    assert "junk" in cols["train"] and any(f["name"] == "shop" for f in cols["features"])
    assert client.patch(f"/api/challenges/{cid}/spec", json={"drop_cols": ["target"]}).status_code == 400
    patched = client.patch(f"/api/challenges/{cid}/spec", json={"id_col": "row_id", "drop_cols": ["junk"]})
    assert patched.status_code == 200 and patched.json()["spec"]["drop_cols"] == ["junk"]
    _wait_idle(client, cid)
    board = client.get(f"/api/challenges/{cid}/leaderboard").json()
    assert board["baselines_stale"] is True and board["entries"] == []
    assert client.get(f"/api/challenges/{cid}/experiments").json()[0]["stale"] is True

    assert client.delete(f"/api/challenges/{cid}/experiments/{exp_id}").status_code == 204
    assert client.get(f"/api/challenges/{cid}/experiments").json() == []


def test_advisor_endpoints_with_a_fake_claude(client, monkeypatch):
    import quest_core.advisor as adv
    from tests.test_advisor import CLOSENESS_CODE, FakeClaude, _p

    cid = client.post("/api/challenges", data={"name": "Ratio", "target": "target", "id_col": "row_id"},
                      files=_ratio_files()).json()["id"]
    _wait_idle(client, cid)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert client.get("/api/advisor").json()["configured"] is False
    client.post(f"/api/challenges/{cid}/jobs", json={"kind": "baselines"})
    _wait_idle(client, cid)
    assert client.post(f"/api/challenges/{cid}/advice", json={}).status_code == 409  # no key

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    models = [m["key"] for m in client.get(f"/api/challenges/{cid}/reports/baselines").json()["models"]
              if m["status"] == "ok" and m["key"] != "dummy"]
    fake = FakeClaude({"reading": "r", "proposals": [
        _p("Closeness in code", 1, [], code=CLOSENESS_CODE, model=models[0]),
        _p("Closeness formula", 2, [{"type": "formula", "params": {"name": "c", "expression": "abs(log(a / b))"}}], model=models[0]),
    ]})
    monkeypatch.setattr(adv, "http_transport", fake)
    monkeypatch.setattr(adv.run_advice, "__defaults__", (None, fake))

    r = client.post(f"/api/challenges/{cid}/advice", json={"focus": "ratios", "n": 3})
    assert r.status_code == 202, r.text
    sid = r.json()["id"]
    _wait_idle(client, cid)
    session = client.get(f"/api/challenges/{cid}/advice").json()[0]
    assert session["job"]["status"] == "done" and "description" not in session
    code_p, formula_p = session["proposals"]
    base = f"/api/challenges/{cid}/advice/{sid}/proposals"
    assert client.post(f"{base}/{code_p['id']}/accept", json={}).status_code == 400  # code not reviewed
    accepted = client.post(f"{base}/{code_p['id']}/accept", json={"code_reviewed": True})
    assert accepted.status_code == 201, accepted.text
    assert accepted.json()["source"] == {"advice": sid, "proposal": code_p["id"]}
    assert client.post(f"{base}/{formula_p['id']}/dismiss").json()["status"] == "dismissed"
    _wait_idle(client, cid)
    exp = client.get(f"/api/challenges/{cid}/experiments").json()[0]
    assert exp["code"] and exp["result"]["vs_control"]["verdict"] == "better"


def test_endgame_endpoints(client):
    cid = client.post("/api/challenges", data={"name": "Ratio", "target": "target", "id_col": "row_id"},
                      files=_ratio_files()).json()["id"]
    _wait_idle(client, cid)
    client.post(f"/api/challenges/{cid}/jobs", json={"kind": "baselines"})
    _wait_idle(client, cid)
    assert client.put(f"/api/challenges/{cid}/endgame/settings", json={"public_share": 1.5}).status_code == 400
    assert client.put(f"/api/challenges/{cid}/endgame/settings", json={"public_share": 0.3, "n_picks": 2}).status_code == 200
    summary = client.get(f"/api/challenges/{cid}/endgame").json()
    assert summary["ready"] and summary["noise"]["n_public"] == 30
    ids = [c["id"] for c in summary["candidates"]]
    r = client.post(f"/api/challenges/{cid}/endgame/blends", json={"name": "Top two", "members": ids[:2], "method": "greedy"})
    assert r.status_code == 201, r.text
    _wait_idle(client, cid)
    summary = client.get(f"/api/challenges/{cid}/endgame").json()
    blend = summary["blends"][0]
    assert blend["job"]["status"] == "done" and blend["result"]["status"] == "ok"
    assert client.get(f"/api/challenges/{cid}/endgame/blends/{blend['id']}/submission").status_code == 200
    assert client.post(f"/api/challenges/{cid}/endgame/submissions", json={"entry_id": "nope", "public_score": 0.9}).status_code == 400
    sub = client.post(f"/api/challenges/{cid}/endgame/submissions", json={"entry_id": ids[0], "public_score": 0.9, "note": "first"})
    assert sub.status_code == 201 and sub.json()["cv_snapshot"] is not None
    assert client.get(f"/api/challenges/{cid}/endgame").json()["tracking"]["rows"][0]["note"] == "first"
