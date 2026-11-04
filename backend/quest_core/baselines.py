"""Baseline model zoo run on the shared CV splits.

Every model sees identical splits, so differences between models are paired. The comparison
against the leader uses the Nadeau-Bengio corrected variance, which accounts for training sets
overlapping across folds and repeats; a naive std/sqrt(n) badly overstates confidence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .cv_designer import adversarial_holdout_split
from .spec import Metric


@dataclass
class ModelSpec:
    key: str
    label: str
    family: str  # "tree" (ordinal cats), "linear" (one-hot + scaling), "native" (model handles strings)
    build: Callable[[], object]


def _leaf_size(n: int) -> int:
    return int(np.clip(n // 50, 5, 20))


def available_models(task: str, n: int, seed: int) -> list[ModelSpec]:
    clf = task != "regression"
    leaf = _leaf_size(n)
    k_neighbors = int(np.clip(np.sqrt(n) / 2, 5, 50))
    specs = [
        ModelSpec("dummy", "Prior / mean", "tree",
                  lambda: DummyClassifier(strategy="prior") if clf else DummyRegressor()),
        ModelSpec("linear", "Logistic regression" if clf else "Ridge", "linear",
                  lambda: LogisticRegression(C=1.0, max_iter=3000) if clf else Ridge(alpha=1.0)),
        ModelSpec("knn", f"k-NN (k={k_neighbors})", "linear",
                  lambda: KNeighborsClassifier(k_neighbors) if clf else KNeighborsRegressor(k_neighbors)),
        ModelSpec("random_forest", "Random forest", "tree",
                  lambda: (RandomForestClassifier if clf else RandomForestRegressor)(
                      n_estimators=300, min_samples_leaf=2, max_features="sqrt" if clf else 0.5,
                      n_jobs=-1, random_state=seed)),
        ModelSpec("hist_gb", "Histogram GBM (sklearn)", "tree",
                  lambda: (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(
                      max_iter=400, learning_rate=0.05, min_samples_leaf=leaf, l2_regularization=1.0,
                      early_stopping=False, random_state=seed)),
    ]
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        specs.append(ModelSpec("lightgbm", "LightGBM", "tree",
                               lambda: (LGBMClassifier if clf else LGBMRegressor)(
                                   n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=leaf,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                   random_state=seed, verbose=-1)))
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        loss = {"binary": "Logloss", "multiclass": "MultiClass", "regression": "RMSE"}[task]
        specs.append(ModelSpec("catboost", "CatBoost", "native",
                               lambda: (CatBoostClassifier if clf else CatBoostRegressor)(
                                   iterations=1000, learning_rate=0.05, depth=6, loss_function=loss,
                                   random_seed=seed, verbose=0, allow_writing_files=False)))
    except ImportError:
        pass
    return specs


def _preprocessor(family: str) -> ColumnTransformer:
    """Columns are picked by dtype at fit time, so feature recipes can add or drop columns freely."""
    num_cols = make_column_selector(dtype_exclude=object)
    cat_cols = make_column_selector(dtype_include=object)
    if family == "linear":
        num = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
        cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, max_categories=30)
    else:
        num = "passthrough"
        cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)], remainder="drop")


class _NativeCategorical(BaseEstimator):
    """Adapter so CatBoost receives raw string categoricals through the same fit/predict calls."""

    def __init__(self, model=None, cat_cols=None):
        self.model = model
        self.cat_cols = cat_cols

    def fit(self, X, y):
        cats = self.cat_cols if self.cat_cols is not None else [c for c in X.columns if X[c].dtype == object]
        self.model.fit(X, y, cat_features=cats)
        self.classes_ = getattr(self.model, "classes_", None)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def build_estimator(spec: ModelSpec, task: str, metric: Metric, recipe=None):
    """Model pipeline: optional feature recipe -> preprocessing -> model (log-target wrapper for RMSLE)."""
    if spec.family == "native":
        est = _NativeCategorical(spec.build())
    else:
        est = Pipeline([("prep", _preprocessor(spec.family)), ("model", spec.build())])
    if recipe is not None:
        est = Pipeline([("features", recipe), ("estimator", est)])
    if task == "regression" and metric.name == "rmsle":
        est = TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
    return est


def _predict(est, X: pd.DataFrame, task: str, n_classes: int) -> np.ndarray:
    if task == "regression":
        return np.asarray(est.predict(X), dtype=float).ravel()
    proba = np.asarray(est.predict_proba(X), dtype=float)
    classes = np.asarray(est.classes_).astype(int)
    out = np.zeros((len(X), n_classes))
    out[:, classes] = proba  # a fold may lack a class (time splits); keep columns aligned
    return out


def _safe_score(metric: Metric, y: np.ndarray, pred: np.ndarray) -> float:
    try:
        return float(metric.fn(y, pred))
    except ValueError:  # e.g. AUC on a fold containing a single class
        return float("nan")


def corrected_comparison(diffs: np.ndarray, n_splits: int) -> dict:
    """Nadeau-Bengio corrected paired comparison over CV folds."""
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return {"mean_diff": float(diffs.mean()) if len(diffs) else None, "corrected_se": None, "within_noise": True}
    test_train_ratio = 1 / max(n_splits - 1, 1)
    se = round(float(np.sqrt((1 / len(diffs) + test_train_ratio) * diffs.var(ddof=1))), 12)
    mean = round(float(diffs.mean()), 12)
    return {"mean_diff": mean, "corrected_se": se, "within_noise": bool(abs(mean) <= 2 * se) if se > 0 else mean == 0}


def evaluate(
    make_estimator: Callable[[], object],
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    task: str,
    metric: Metric,
    splits: list[dict],
    holdout: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Cross-validate one estimator on fixed splits, score the test-like holdout, then fit on all rows.

    Shared by baselines and experiments so both are measured by exactly the same code.
    """
    n_classes = int(y.max()) + 1 if task != "regression" else 1
    started = time.perf_counter()
    shape = (len(X), n_classes) if task != "regression" else (len(X),)
    oof_sum, oof_count = np.zeros(shape), np.zeros(len(X))
    fold_scores = []
    for split in splits:
        tr, va = split["train"], split["valid"]
        est = make_estimator().fit(X.iloc[tr], y[tr])
        pred = _predict(est, X.iloc[va], task, n_classes)
        fold_scores.append(_safe_score(metric, y[va], pred))
        oof_sum[va] += pred
        oof_count[va] += 1

    covered = oof_count > 0
    oof = np.full(shape, np.nan)
    oof[covered] = oof_sum[covered] / (oof_count[covered] if task == "regression" else oof_count[covered, None])

    holdout_score = None
    if holdout is not None:
        h_tr, h_va = holdout
        est = make_estimator().fit(X.iloc[h_tr], y[h_tr])
        holdout_score = _safe_score(metric, y[h_va], _predict(est, X.iloc[h_va], task, n_classes))

    final = make_estimator().fit(X, y)
    test_pred = _predict(final, X_test, task, n_classes)

    scores = np.asarray(fold_scores, dtype=float)
    finite = scores[np.isfinite(scores)]
    result = {
        "status": "ok",
        "cv_mean": float(finite.mean()) if len(finite) else None,
        "cv_std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "fold_scores": [None if not np.isfinite(v) else float(v) for v in scores],
        "oof_score": _safe_score(metric, y[covered], oof[covered]),
        "holdout_score": holdout_score,
        "seconds": round(time.perf_counter() - started, 2),
    }
    return result, {"oof": oof, "test": test_pred, "fold_scores": scores, "final": final}


def model_spec(key: str, task: str, n: int, seed: int) -> ModelSpec:
    for spec in available_models(task, n, seed):
        if spec.key == key:
            return spec
    raise ValueError(f"Model '{key}' isn't available here. Install it or pick another.")


def holdout_for(plan: dict, likeness: np.ndarray | None):
    if plan.get("adversarial_holdout") and likeness is not None:
        return adversarial_holdout_split(likeness, plan.get("holdout_share") or 0.2)
    return None


def run_baselines(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    task: str,
    metric: Metric,
    splits: list[dict],
    plan: dict,
    likeness: np.ndarray | None = None,
    seed: int = 42,
    only: list[str] | None = None,
    on_model: Callable[[str], None] | None = None,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Returns the JSON-ready report and per-model arrays {"oof", "test"} for saving."""
    specs = [s for s in available_models(task, len(X), seed) if not only or s.key in only]
    holdout = holdout_for(plan, likeness)

    results, arrays = [], {}
    for spec in specs:
        if on_model:
            on_model(spec.label)
        try:
            result, arr = evaluate(lambda: build_estimator(spec, task, metric), X, y, X_test, task, metric,
                                   splits, holdout)
        except Exception as exc:  # one broken model must not sink the whole run
            results.append({"key": spec.key, "label": spec.label, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            continue
        results.append({"key": spec.key, "label": spec.label, **result})
        arr.pop("final")
        arrays[spec.key] = arr

    _rank(results, arrays, metric, plan["n_splits"])
    return {
        "metric": metric.name,
        "metric_label": metric.label,
        "greater_is_better": metric.greater_is_better,
        "n_folds": len(splits),
        "holdout_rows": int(len(holdout[1])) if holdout is not None else None,
        "models": results,
    }, arrays


def _rank(results: list[dict], arrays: dict, metric: Metric, n_splits: int) -> None:
    ok = [r for r in results if r["status"] == "ok" and r["cv_mean"] is not None]
    if not ok:
        return
    sign = 1 if metric.greater_is_better else -1
    ok.sort(key=lambda r: sign * r["cv_mean"], reverse=True)
    leader = ok[0]
    leader_scores = arrays[leader["key"]]["fold_scores"]
    for rank, r in enumerate(ok, start=1):
        r["rank"] = rank
        if r is leader:
            r["vs_leader"] = {"mean_diff": 0.0, "corrected_se": None, "within_noise": True}
            continue
        diffs = sign * (arrays[r["key"]]["fold_scores"] - leader_scores)  # negative = worse than leader
        r["vs_leader"] = corrected_comparison(diffs, n_splits)
    results.sort(key=lambda r: r.get("rank", 10_000))


def build_submission(
    ids: pd.Series,
    pred: np.ndarray,
    task: str,
    metric: Metric,
    class_labels: list,
    target_name: str,
    id_name: str,
) -> pd.DataFrame:
    """A starting submission file. Check the challenge's required format before submitting."""
    frame = pd.DataFrame({id_name: ids.to_numpy()})
    if task == "regression":
        frame[target_name] = pred
    elif metric.uses_labels:
        frame[target_name] = np.asarray(class_labels, dtype=object)[pred.argmax(axis=1)]
    elif task == "binary":
        frame[target_name] = pred[:, 1]
    else:
        for i, label in enumerate(class_labels):
            frame[str(label)] = pred[:, i]
    return frame
