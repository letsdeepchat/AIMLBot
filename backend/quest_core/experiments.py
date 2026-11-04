"""Feature experiments and the combined leaderboard.

An experiment is a feature recipe plus one model from the baseline zoo. It runs on exactly the
splits the baselines used, so its fold scores pair with its control (the same model without the
recipe) and with every other entry on the leaderboard.

Every result carries the fingerprint of the settings that produced it. Results from other settings
are reported as stale instead of being compared against incompatible folds.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import baselines as bl
from .cv_designer import make_splits
from .pipeline import _write_json, load_inputs
from .recipes import FeatureRecipe, validate_recipe
from .sandbox import SandboxError, run_features
from .spec import ChallengeSpec
from .tracking import get_tracker
from .univariate import univariate_strength

Stage = Callable[[str], None]
MAX_SCORED_NEW_COLUMNS = 30


def experiments_root(challenge_dir: Path) -> Path:
    return Path(challenge_dir) / "experiments"


def _exp_dir(challenge_dir: Path, exp_id: str) -> Path:
    if not exp_id.isalnum():
        raise ValueError("Bad experiment id.")
    return experiments_root(challenge_dir) / exp_id


def feature_kinds(analysis: dict) -> dict[str, str]:
    used = set(analysis["features"]["used"])
    return {c["name"]: c["kind"] for c in analysis["profile"]["columns"] if c["name"] in used}


def n_classes_of(analysis: dict) -> int:
    return len(analysis["profile"]["target"].get("classes") or []) or 2


def create_experiment(challenge_dir: Path, name: str, model: str, steps: list[dict], hypothesis: str = "",
                      parent: str | None = None, code: str | None = None, source: dict | None = None) -> dict:
    exp_id = uuid.uuid4().hex[:10]
    definition = {
        "id": exp_id, "name": name.strip() or "Untitled experiment", "hypothesis": hypothesis.strip(),
        "model": model, "steps": steps, "code": (code or "").strip() or None, "parent": parent,
        "source": source, "created_at": time.time(), "result": None,
    }
    folder = _exp_dir(challenge_dir, exp_id)
    folder.mkdir(parents=True)
    _write_json(folder / "experiment.json", definition)
    return definition


def load_experiment(challenge_dir: Path, exp_id: str) -> dict:
    path = _exp_dir(challenge_dir, exp_id) / "experiment.json"
    if not path.exists():
        raise FileNotFoundError(f"No experiment '{exp_id}'.")
    return json.loads(path.read_text())


def list_experiments(challenge_dir: Path) -> list[dict]:
    root = experiments_root(challenge_dir)
    if not root.exists():
        return []
    items = [json.loads(p.read_text()) for p in root.glob("*/experiment.json")]
    return sorted(items, key=lambda e: e["created_at"], reverse=True)


def delete_experiment(challenge_dir: Path, exp_id: str) -> None:
    shutil.rmtree(_exp_dir(challenge_dir, exp_id), ignore_errors=True)


def submission_path(challenge_dir: Path, exp_id: str) -> Path:
    return _exp_dir(challenge_dir, exp_id) / "submission.csv"


def code_input(inp, spec: ChallengeSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw columns handed to feature code: every feature column (text included) plus the time column.
    Never the target, so code can't build features from the label."""
    cols = list(inp.features)
    if spec.time_col and spec.time_col in inp.test.columns:
        cols.append(spec.time_col)
    return inp.train[cols], inp.test[cols]


_code_cache: dict[str, tuple[dict, str | None]] = {}


def code_kinds(challenge_dir: Path, spec: ChallengeSpec, code: str, sample: int = 300) -> tuple[dict[str, str], str | None]:
    """Run code on a sample of train rows to learn its output columns. Cached per challenge and code."""
    key = hashlib.sha1(f"{challenge_dir}|{spec.drop_cols}|{code}".encode()).hexdigest()
    if key not in _code_cache:
        inp = load_inputs(Path(challenge_dir), spec)
        train_raw, _ = code_input(inp, spec)
        try:
            result = run_features(code, train_raw.iloc[:sample], timeout=60)
            _code_cache[key] = (result.columns, None)
        except SandboxError as exc:
            _code_cache[key] = ({}, str(exc))
        if len(_code_cache) > 64:
            _code_cache.pop(next(iter(_code_cache)))
    return _code_cache[key]


def _verdict(comparison: dict) -> str:
    if comparison["mean_diff"] is None or comparison["within_noise"]:
        return "no_clear_change"
    return "better" if comparison["mean_diff"] > 0 else "worse"


def _fold_array(values: list) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _new_column_strength(recipe: FeatureRecipe, X: pd.DataFrame, y: np.ndarray, task: str, seed: int,
                         base_columns: list[str]) -> list[dict]:
    transformed = recipe.fit_transform(X, y)  # target encoding is cross-fitted here, so no leak
    new_cols = [c for c in transformed.columns if c not in base_columns][:MAX_SCORED_NEW_COLUMNS]
    rows = []
    for c in new_cols:
        values = transformed[c]
        categorical = values.dtype == object
        x = pd.factorize(values)[0].astype(float) if categorical else values.to_numpy(dtype=float)
        strength = univariate_strength(x, y, task, seed, categorical=categorical)
        rows.append({"column": c, "strength": strength})
    return sorted(rows, key=lambda r: r["strength"] if r["strength"] is not None else -1, reverse=True)


def run_experiment(challenge_dir: Path, spec: ChallengeSpec, exp_id: str, on_stage: Stage | None = None) -> dict:
    stage = on_stage or (lambda _: None)
    challenge_dir = Path(challenge_dir)
    folder = _exp_dir(challenge_dir, exp_id)
    experiment = load_experiment(challenge_dir, exp_id)

    analysis_path, baselines_path = challenge_dir / "analysis.json", challenge_dir / "baselines.json"
    if not baselines_path.exists():
        raise RuntimeError("Run the baselines first. Each experiment is compared against its baseline model.")
    analysis = json.loads(analysis_path.read_text())
    baselines = json.loads(baselines_path.read_text())
    fingerprint = analysis.get("fingerprint")
    if baselines.get("fingerprint") != fingerprint:
        raise RuntimeError("The baselines were run with different settings. Re-run them, then this experiment.")

    stage("Loading data")
    inp = load_inputs(challenge_dir, spec)
    X, X_test, kinds = inp.X, inp.X_test, feature_kinds(analysis)
    if experiment.get("code"):
        stage("Running feature code in the sandbox")
        train_raw, test_raw = code_input(inp, spec)
        produced = run_features(experiment["code"], train_raw, test_raw)
        X = pd.concat([inp.X, produced.train.set_index(inp.X.index)], axis=1)
        X_test = pd.concat([inp.X_test, produced.test.set_index(inp.X_test.index)], axis=1)
        kinds = {**kinds, **produced.columns}

    check = validate_recipe(experiment["steps"], kinds, analysis["task"], n_classes_of(analysis))
    if check["errors"]:
        first = check["errors"][0]
        raise ValueError(f"Step {first['step'] + 1}: {first['message']}")
    plan = analysis["cv_plan"]
    splits = make_splits(plan, inp.y, inp.groups,
                         inp.time_train.to_numpy() if inp.time_train is not None else None, spec.seed)
    likeness_path = challenge_dir / "likeness.npy"
    holdout = bl.holdout_for(plan, np.load(likeness_path) if likeness_path.exists() else None)
    model = bl.model_spec(experiment["model"], inp.task, len(inp.X), spec.seed)

    n_steps = len(experiment["steps"])
    stage(f"Cross-validating {model.label} with {n_steps} feature step{'s' if n_steps != 1 else ''}"
          + (" and feature code" if experiment.get("code") else ""))
    result, arrays = bl.evaluate(
        lambda: bl.build_estimator(model, inp.task, inp.metric, FeatureRecipe(experiment["steps"], inp.task, spec.seed)),
        X, inp.y, X_test, inp.task, inp.metric, splits, holdout,
    )

    sign = 1 if inp.metric.greater_is_better else -1
    control = next((m for m in baselines["models"] if m["key"] == experiment["model"] and m["status"] == "ok"), None)
    vs_control = None
    if control:
        diffs = sign * (arrays["fold_scores"] - _fold_array(control["fold_scores"]))  # positive = recipe helps
        cmp = bl.corrected_comparison(diffs, plan["n_splits"])
        vs_control = {
            **cmp, "verdict": _verdict(cmp), "control_cv_mean": control["cv_mean"],
            "holdout_diff": (sign * (result["holdout_score"] - control["holdout_score"])
                             if result["holdout_score"] is not None and control.get("holdout_score") is not None else None),
        }

    stage("Scoring the new columns")
    new_columns = _new_column_strength(FeatureRecipe(experiment["steps"], inp.task, spec.seed), X, inp.y,
                                       inp.task, spec.seed, list(inp.X.columns))

    stage("Writing predictions")
    np.save(folder / "oof.npy", arrays["oof"])
    np.save(folder / "test.npy", arrays["test"])
    ids = inp.test[spec.id_col] if spec.id_col and spec.id_col in inp.test.columns else pd.Series(range(len(inp.test)))
    bl.build_submission(ids, arrays["test"], inp.task, inp.metric, inp.class_labels, spec.target,
                        spec.id_col or "id").to_csv(folder / "submission.csv", index=False)

    experiment["result"] = {
        **result, "model_label": model.label, "vs_control": vs_control, "new_columns": new_columns,
        "n_output_columns": len(check["output"]), "finished_at": time.time(),
    }
    experiment["fingerprint"] = fingerprint
    _write_json(folder / "experiment.json", experiment)

    get_tracker(challenge_dir / "runs.jsonl").log_run(
        experiment=f"quest-{challenge_dir.name}", name=f"experiment-{experiment['name']}",
        params={"model": experiment["model"], "steps": json.dumps(experiment["steps"])[:480],
                "n_steps": len(experiment["steps"]), "cv_strategy": plan["strategy"]},
        metrics={"cv_mean": result["cv_mean"], "cv_std": result["cv_std"], "oof_score": result["oof_score"],
                 "holdout_score": result["holdout_score"],
                 "diff_vs_control": vs_control["mean_diff"] if vs_control else None},
        tags={"stage": "experiment", "experiment_id": exp_id, "verdict": vs_control["verdict"] if vs_control else "none"},
        artifacts=[folder / "submission.csv"],
    )
    return experiment


def build_leaderboard(challenge_dir: Path) -> dict:
    """Baselines and finished experiments on one scale, ranked, each paired against the leader."""
    challenge_dir = Path(challenge_dir)
    analysis_path, baselines_path = challenge_dir / "analysis.json", challenge_dir / "baselines.json"
    if not analysis_path.exists() or not baselines_path.exists():
        return {"entries": [], "stale": [], "ready": False}
    analysis = json.loads(analysis_path.read_text())
    baselines = json.loads(baselines_path.read_text())
    fingerprint = analysis.get("fingerprint")
    plan = analysis["cv_plan"]
    sign = 1 if baselines["greater_is_better"] else -1

    entries, stale = [], []
    baselines_current = baselines.get("fingerprint") == fingerprint
    for m in baselines["models"]:
        if m["status"] != "ok":
            continue
        entry = {"id": f"baseline:{m['key']}", "kind": "baseline", "key": m["key"], "label": m["label"],
                 "model": m["key"], "cv_mean": m["cv_mean"], "cv_std": m["cv_std"],
                 "holdout_score": m.get("holdout_score"), "fold_scores": m["fold_scores"]}
        (entries if baselines_current else stale).append(entry)
    for e in list_experiments(challenge_dir):
        r = e.get("result")
        if not r or r.get("status") != "ok":
            continue
        entry = {"id": f"experiment:{e['id']}", "kind": "experiment", "key": e["id"], "label": e["name"],
                 "model": e["model"], "model_label": r.get("model_label"), "cv_mean": r["cv_mean"], "cv_std": r["cv_std"],
                 "holdout_score": r.get("holdout_score"), "fold_scores": r["fold_scores"],
                 "verdict": (r.get("vs_control") or {}).get("verdict")}
        (entries if e.get("fingerprint") == fingerprint else stale).append(entry)

    entries = [e for e in entries if e["cv_mean"] is not None]
    entries.sort(key=lambda e: sign * e["cv_mean"], reverse=True)
    if entries:
        leader_scores = _fold_array(entries[0]["fold_scores"])
        for rank, e in enumerate(entries, start=1):
            e["rank"] = rank
            if rank == 1:
                e["vs_leader"] = {"mean_diff": 0.0, "corrected_se": None, "within_noise": True}
            else:
                diffs = sign * (_fold_array(e["fold_scores"]) - leader_scores)
                e["vs_leader"] = bl.corrected_comparison(diffs, plan["n_splits"])

    holdout_leader = None
    with_holdout = [e for e in entries if e.get("holdout_score") is not None]
    if with_holdout:
        holdout_leader = max(with_holdout, key=lambda e: sign * e["holdout_score"])["id"]

    for e in entries + stale:
        e.pop("fold_scores", None)
    return {
        "ready": True, "metric_label": baselines["metric_label"], "greater_is_better": baselines["greater_is_better"],
        "n_folds": baselines["n_folds"], "baselines_stale": not baselines_current,
        "holdout_leader": holdout_leader, "entries": entries, "stale": stale,
    }
