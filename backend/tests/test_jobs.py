"""Store and job-runner tests. They use the inline backend, so no Redis is needed."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from tests.test_core import _binary_data


def _wait(store, job_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get_job(job_id)
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.2)
    raise TimeoutError(f"Job {job_id} did not finish.")


def test_inline_jobs_run_analysis_then_baselines():
    with tempfile.TemporaryDirectory() as d:
        os.environ["QUEST_DATA_DIR"] = d
        os.environ["QUEST_JOB_BACKEND"] = "inline"
        os.environ.pop("QUEST_DB_PATH", None)
        from app.config import load_settings
        from app.db import Store
        from app.jobs import JobRejected, challenge_dir, enqueue

        settings = load_settings()
        store = Store(settings.db_path)
        train, test = _binary_data(n=300, n_test=120)
        challenge = store.create_challenge(
            name="Synthetic", description="", task="binary", spec={"target": "target", "id_col": "row_id",
                                                                    "metric": "roc_auc", "task": "binary"},
            submission_limit=None, deadline=None, train_file="train.csv", test_file="test.csv",
            n_train=len(train), n_test=len(test),
        )
        root = challenge_dir(settings, challenge["id"])
        root.mkdir(parents=True)
        train.to_csv(root / "train.csv", index=False)
        test.to_csv(root / "test.csv", index=False)

        try:
            enqueue(settings, store, challenge["id"], "baselines")
            raise AssertionError("baselines must wait for the analysis")
        except JobRejected:
            pass

        job = enqueue(settings, store, challenge["id"], "analysis")
        assert enqueue(settings, store, challenge["id"], "analysis")["id"] == job["id"]  # idempotent
        done = _wait(store, job["id"])
        assert done["status"] == "done", done["error"]
        assert (root / "analysis.json").exists()

        job = enqueue(settings, store, challenge["id"], "baselines")
        done = _wait(store, job["id"])
        assert done["status"] == "done", done["error"]
        assert any(Path(root / "submissions").iterdir())

        # A broken spec surfaces as a readable failure, not a crash.
        bad = store.create_challenge(
            name="Broken", description="", task="binary", spec={"target": "missing_column"},
            submission_limit=None, deadline=None, train_file="train.csv", test_file="test.csv", n_train=1, n_test=1,
        )
        bad_root = challenge_dir(settings, bad["id"])
        bad_root.mkdir(parents=True)
        train.to_csv(bad_root / "train.csv", index=False)
        test.to_csv(bad_root / "test.csv", index=False)
        failed = _wait(store, enqueue(settings, store, bad["id"], "analysis")["id"])
        assert failed["status"] == "failed" and "missing_column" in failed["error"]

        store.delete_challenge(challenge["id"])
        assert store.get_challenge(challenge["id"]) is None
        assert store.jobs_for(challenge["id"]) == []  # cascade


def test_experiment_job_and_week1_database_migration():
    import sqlite3

    with tempfile.TemporaryDirectory() as d:
        # A week-1 database has no jobs.ref column; opening it must add one without losing rows.
        old = Path(d) / "quest.db"
        with sqlite3.connect(old) as conn:
            conn.executescript(
                "CREATE TABLE jobs (id TEXT PRIMARY KEY, challenge_id TEXT, kind TEXT, status TEXT, stage TEXT, "
                "error TEXT, created_at REAL, started_at REAL, finished_at REAL);"
                "INSERT INTO jobs (id, challenge_id, kind, status, created_at) VALUES ('old', 'c', 'analysis', 'done', 0);"
            )
        os.environ["QUEST_DATA_DIR"] = d
        os.environ["QUEST_JOB_BACKEND"] = "inline"
        from app.config import load_settings
        from app.db import Store
        from app.jobs import JobRejected, challenge_dir, enqueue
        from quest_core.experiments import create_experiment, load_experiment
        from tests.test_experiments import _ratio_data

        settings = load_settings()
        store = Store(settings.db_path)
        assert store.get_job("old")["ref"] is None

        train, test = _ratio_data(n=300, n_test=100)
        challenge = store.create_challenge(
            name="Ratio", description="", task="binary",
            spec={"target": "target", "id_col": "row_id", "metric": "roc_auc", "task": "binary"},
            submission_limit=None, deadline=None, train_file="train.csv", test_file="test.csv",
            n_train=len(train), n_test=len(test),
        )
        root = challenge_dir(settings, challenge["id"])
        root.mkdir(parents=True)
        train.to_csv(root / "train.csv", index=False)
        test.to_csv(root / "test.csv", index=False)
        for kind in ("analysis", "baselines"):
            assert _wait(store, enqueue(settings, store, challenge["id"], kind)["id"])["status"] == "done"

        exp = create_experiment(root, "Closeness", "linear",
                                [{"type": "formula", "params": {"name": "c", "expression": "abs(log(a / b))"}}])
        job = enqueue(settings, store, challenge["id"], "experiment", exp["id"])
        assert job["ref"] == exp["id"]
        try:
            enqueue(settings, store, challenge["id"], "baselines")
            raise AssertionError("baselines must not start while an experiment runs")
        except JobRejected:
            pass
        done = _wait(store, job["id"])
        assert done["status"] == "done", done["error"]
        assert load_experiment(root, exp["id"])["result"]["vs_control"]["verdict"] == "better"
        assert store.latest_job_by_ref(challenge["id"])[exp["id"]]["status"] == "done"

        store.update_spec(challenge["id"], {**challenge["spec"], "drop_cols": ["junk"]})
        assert store.get_challenge(challenge["id"])["spec"]["drop_cols"] == ["junk"]
