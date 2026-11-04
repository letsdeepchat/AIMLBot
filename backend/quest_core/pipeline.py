"""Week-1 pipeline: analysis (profile, leakage, shift, CV plan) then baselines.

On-disk layout of one challenge directory:
    train.<ext>, test.<ext>      uploaded files
    analysis.json                profile, leakage, shift and the CV plan
    likeness.npy                 out-of-fold test-likeness of every train row
    baselines.json               model comparison
    preds/<model>_{oof,test}.npy predictions, kept for blending later
    submissions/<model>.csv      starting submission files
    runs.jsonl                   local tracking log when MLflow isn't configured
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import baselines as bl
from .cv_designer import design_cv, make_splits
from .encoding import categorical_columns, column_kinds, jsonable, ordinal_codes, prepare_features
from .leakage import check_leakage
from .profiling import profile_dataset
from .shift import analyze_shift
from .spec import ChallengeSpec, Metric, feature_columns, infer_task, resolve_metric
from .tracking import get_tracker

Stage = Callable[[str], None]
DATA_SUFFIXES = (".csv", ".tsv", ".parquet", ".pq")


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type '{suffix}'. Upload CSV, TSV or Parquet.")


def find_file(challenge_dir: Path, stem: str) -> Path:
    for suffix in DATA_SUFFIXES:
        candidate = challenge_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {stem} file found in {challenge_dir}.")


@dataclass
class Inputs:
    train: pd.DataFrame
    test: pd.DataFrame
    task: str
    metric: Metric
    features: list[str]
    kinds: dict[str, str]
    X: pd.DataFrame
    X_test: pd.DataFrame
    X_codes: pd.DataFrame
    X_test_codes: pd.DataFrame
    y: np.ndarray
    class_labels: list
    groups: np.ndarray | None
    time_train: pd.Series | None
    time_test: pd.Series | None


def _time_values(df: pd.DataFrame, col: str) -> pd.Series:
    prepared = prepare_features(df[[col]], {col: "datetime" if not pd.api.types.is_numeric_dtype(df[col]) else "numeric"})
    return prepared[col]


def load_inputs(challenge_dir: Path, spec: ChallengeSpec) -> Inputs:
    train = load_table(find_file(challenge_dir, "train"))
    test = load_table(find_file(challenge_dir, "test"))
    if spec.target not in train.columns:
        raise ValueError(f"Target column '{spec.target}' is not in the train file.")
    train = train[train[spec.target].notna()].reset_index(drop=True)

    task = spec.task or infer_task(train[spec.target])
    metric = resolve_metric(task, spec.metric)
    features = feature_columns(train, test, spec)
    if not features:
        raise ValueError("No feature columns are shared by train and test after exclusions.")
    kinds = column_kinds(train, features)

    X = prepare_features(train, kinds)
    X_test = prepare_features(test, kinds)
    X_codes, X_test_codes = ordinal_codes(X, X_test)

    if task == "regression":
        y = train[spec.target].astype(float).to_numpy()
        class_labels: list = []
    else:
        labels, y = np.unique(train[spec.target].to_numpy(dtype=object).astype(str), return_inverse=True)
        class_labels = labels.tolist()
        # Restore numeric labels (0/1) so submission files keep the original type.
        try:
            numeric_labels = [float(v) for v in class_labels]
            class_labels = [int(v) if v.is_integer() else v for v in numeric_labels]
        except ValueError:
            pass

    groups = train[spec.group_col].astype(str).to_numpy() if spec.group_col else None
    time_train = _time_values(train, spec.time_col) if spec.time_col else None
    time_test = _time_values(test, spec.time_col) if spec.time_col and spec.time_col in test.columns else None
    return Inputs(train, test, task, metric, features, kinds, X, X_test, X_codes, X_test_codes,
                  np.asarray(y), class_labels, groups, time_train, time_test)


def plan_fingerprint(plan: dict, spec: ChallengeSpec, n_train: int) -> str:
    """Identifies settings whose scores are comparable: same rows, folds, holdout, features and metric."""
    payload = {
        "plan": {k: plan.get(k) for k in ("strategy", "n_splits", "n_repeats", "adversarial_holdout", "holdout_share")},
        "n_train": n_train, "seed": spec.seed, "target": spec.target, "metric": spec.metric,
        "id": spec.id_col, "group": spec.group_col, "time": spec.time_col, "drop": sorted(spec.drop_cols),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2))
    tmp.replace(path)  # atomic: the UI never reads a half-written report


def run_analysis(challenge_dir: Path, spec: ChallengeSpec, on_stage: Stage | None = None) -> dict:
    stage = on_stage or (lambda _: None)
    challenge_dir = Path(challenge_dir)

    stage("Loading data")
    inp = load_inputs(challenge_dir, spec)

    stage("Profiling columns")
    profile = profile_dataset(inp.train, inp.test, spec.target, inp.task, inp.kinds)

    stage("Checking for leakage")
    leakage = check_leakage(inp.train, inp.test, spec.target, inp.task, inp.features, inp.kinds,
                            inp.X_codes, inp.y, spec.group_col, spec.seed)

    stage("Measuring train/test shift")
    time_values = (inp.time_train, inp.time_test) if inp.time_test is not None else None
    shift, likeness = analyze_shift(inp.X, inp.X_test, inp.X_codes, inp.X_test_codes,
                                    categorical_columns(inp.X), spec.seed, time_values)

    stage("Designing validation")
    plan = design_cv(
        inp.task, len(inp.train), profile["target"].get("min_class_count"),
        has_groups=inp.groups is not None, has_time=inp.time_train is not None,
        shift=shift, leakage=leakage,
        n_groups=len(np.unique(inp.groups)) if inp.groups is not None else None,
    )

    report = {
        "generated_at": time.time(),
        "spec": asdict(spec),
        "task": inp.task,
        "metric": {"name": inp.metric.name, "label": inp.metric.label, "greater_is_better": inp.metric.greater_is_better},
        "features": {
            "used": [c for c in inp.features if inp.kinds[c] != "text"],
            "text_excluded": [c for c in inp.features if inp.kinds[c] == "text"],
            "reserved": sorted(spec.excluded()),
        },
        "profile": profile,
        "leakage": leakage,
        "shift": shift,
        "cv_plan": plan,
        "fingerprint": plan_fingerprint(plan, spec, len(inp.train)),
    }
    np.save(challenge_dir / "likeness.npy", likeness)
    _write_json(challenge_dir / "analysis.json", report)

    tracker = get_tracker(challenge_dir / "runs.jsonl")
    tracker.log_run(
        experiment=f"quest-{challenge_dir.name}", name="analysis",
        params={"task": inp.task, "metric": inp.metric.name, "cv_strategy": plan["strategy"],
                "n_splits": plan["n_splits"], "n_repeats": plan["n_repeats"]},
        metrics={"adversarial_auc": shift["adversarial"]["auc"], "n_train": len(inp.train), "n_test": len(inp.test),
                 "high_findings": sum(f["severity"] == "high" for f in leakage["findings"])},
        tags={"stage": "analysis"},
        artifacts=[challenge_dir / "analysis.json"],
    )
    return report


def run_baselines(challenge_dir: Path, spec: ChallengeSpec, on_stage: Stage | None = None,
                  only: list[str] | None = None) -> dict:
    stage = on_stage or (lambda _: None)
    challenge_dir = Path(challenge_dir)
    analysis_path = challenge_dir / "analysis.json"
    if not analysis_path.exists():
        raise RuntimeError("Run the analysis first: baselines use its validation plan.")
    analysis = json.loads(analysis_path.read_text())
    plan = analysis["cv_plan"]

    stage("Loading data")
    inp = load_inputs(challenge_dir, spec)
    splits = make_splits(plan, inp.y, inp.groups,
                         inp.time_train.to_numpy() if inp.time_train is not None else None, spec.seed)
    likeness_path = challenge_dir / "likeness.npy"
    likeness = np.load(likeness_path) if likeness_path.exists() else None

    report, arrays = bl.run_baselines(
        inp.X, inp.y, inp.X_test, inp.task, inp.metric, splits, plan, likeness, spec.seed, only,
        on_model=lambda label: stage(f"Training {label}"),
    )

    stage("Writing predictions")
    preds_dir, subs_dir = challenge_dir / "preds", challenge_dir / "submissions"
    preds_dir.mkdir(exist_ok=True)
    subs_dir.mkdir(exist_ok=True)
    ids = inp.test[spec.id_col] if spec.id_col and spec.id_col in inp.test.columns else pd.Series(range(len(inp.test)))
    id_name = spec.id_col or "id"
    for key, arr in arrays.items():
        np.save(preds_dir / f"{key}_oof.npy", arr["oof"])
        np.save(preds_dir / f"{key}_test.npy", arr["test"])
        sub = bl.build_submission(ids, arr["test"], inp.task, inp.metric, inp.class_labels, spec.target, id_name)
        sub.to_csv(subs_dir / f"{key}.csv", index=False)

    report.update({
        "generated_at": time.time(),
        "task": inp.task,
        "class_labels": inp.class_labels,
        "positive_class": inp.class_labels[1] if inp.task == "binary" else None,
        "cv_strategy": plan["strategy"],
        "submissions": sorted(arrays),
        "fingerprint": analysis.get("fingerprint"),
    })
    _write_json(challenge_dir / "baselines.json", report)

    tracker = get_tracker(challenge_dir / "runs.jsonl")
    for model in report["models"]:
        if model["status"] != "ok":
            continue
        tracker.log_run(
            experiment=f"quest-{challenge_dir.name}", name=f"baseline-{model['key']}",
            params={"model": model["key"], "metric": inp.metric.name, "cv_strategy": plan["strategy"],
                    "n_folds": len(splits)},
            metrics={"cv_mean": model["cv_mean"], "cv_std": model["cv_std"], "oof_score": model["oof_score"],
                     "holdout_score": model["holdout_score"], "seconds": model["seconds"]},
            tags={"stage": "baseline", "rank": str(model.get("rank"))},
            artifacts=[subs_dir / f"{model['key']}.csv"],
        )
    return report
