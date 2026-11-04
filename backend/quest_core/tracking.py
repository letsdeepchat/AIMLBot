"""Experiment tracking.

MLflow is used when it's installed and MLFLOW_TRACKING_URI is set. Otherwise runs are written as
JSON lines next to the challenge data, so nothing is lost and tests need no server.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .encoding import jsonable


class LocalTracker:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_run(self, experiment: str, name: str, params: dict, metrics: dict, tags: dict | None = None,
                artifacts: list[Path] | None = None) -> str:
        run_id = f"local-{int(time.time() * 1000)}"
        record = {"run_id": run_id, "experiment": experiment, "name": name, "params": params,
                  "metrics": metrics, "tags": tags or {}, "artifacts": [str(a) for a in artifacts or []],
                  "logged_at": time.time()}
        with self.path.open("a") as fh:
            fh.write(json.dumps(jsonable(record)) + "\n")
        return run_id


class MLflowTracker:
    def __init__(self, uri: str):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(uri)

    def log_run(self, experiment: str, name: str, params: dict, metrics: dict, tags: dict | None = None,
                artifacts: list[Path] | None = None) -> str:
        mlflow = self.mlflow
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
            mlflow.set_tags(tags or {})
            for artifact in artifacts or []:
                if Path(artifact).exists():
                    mlflow.log_artifact(str(artifact))
            return run.info.run_id


def get_tracker(local_path: Path):
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if uri:
        try:
            return MLflowTracker(uri)
        except ImportError:
            pass
    return LocalTracker(local_path)
