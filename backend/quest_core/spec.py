"""Challenge specification, task inference and metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn import metrics as skm

TASKS = ("binary", "multiclass", "regression")


@dataclass
class ChallengeSpec:
    target: str
    metric: str | None = None
    task: str | None = None
    id_col: str | None = None
    group_col: str | None = None
    time_col: str | None = None
    drop_cols: list[str] = field(default_factory=list)
    seed: int = 42

    def excluded(self) -> set[str]:
        cols = {self.target, self.id_col, self.group_col, self.time_col, *self.drop_cols}
        return {c for c in cols if c}


def infer_task(y: pd.Series) -> str:
    y = y.dropna()
    n_unique = y.nunique()
    if n_unique <= 2:
        return "binary"
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "multiclass"
    is_integer_valued = np.allclose(y.to_numpy(dtype=float), np.round(y.to_numpy(dtype=float)))
    if is_integer_valued and n_unique <= 20 and n_unique / len(y) < 0.05:
        return "multiclass"
    return "regression"


def feature_columns(train: pd.DataFrame, test: pd.DataFrame, spec: ChallengeSpec) -> list[str]:
    """Columns usable as features: present in both files and not reserved by the spec."""
    test_cols = set(test.columns)
    return [c for c in train.columns if c in test_cols and c not in spec.excluded()]


# ---------------------------------------------------------------- metrics


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    greater_is_better: bool
    tasks: tuple[str, ...]
    fn: Callable[[np.ndarray, np.ndarray], float]
    uses_labels: bool = False  # classification metrics scored on argmax, not probabilities


def _positive_proba(pred: np.ndarray) -> np.ndarray:
    return pred[:, 1] if pred.ndim == 2 else pred


def _roc_auc(y, pred):
    if pred.ndim == 2 and pred.shape[1] > 2:
        return skm.roc_auc_score(y, pred, multi_class="ovr", labels=np.arange(pred.shape[1]))
    return skm.roc_auc_score(y, _positive_proba(pred))


def _log_loss(y, pred):
    k = pred.shape[1] if pred.ndim == 2 else 2
    return skm.log_loss(y, pred, labels=np.arange(k))


def _accuracy(y, pred):
    return skm.accuracy_score(y, pred.argmax(axis=1))


def _f1_macro(y, pred):
    return skm.f1_score(y, pred.argmax(axis=1), average="macro")


def _rmse(y, pred):
    return float(np.sqrt(skm.mean_squared_error(y, pred)))


def _rmsle(y, pred):
    return float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(y)) ** 2)))


METRICS: dict[str, Metric] = {
    m.name: m
    for m in [
        Metric("roc_auc", "ROC AUC", True, ("binary", "multiclass"), _roc_auc),
        Metric("log_loss", "Log loss", False, ("binary", "multiclass"), _log_loss),
        Metric("accuracy", "Accuracy", True, ("binary", "multiclass"), _accuracy, uses_labels=True),
        Metric("f1_macro", "Macro F1", True, ("binary", "multiclass"), _f1_macro, uses_labels=True),
        Metric("rmse", "RMSE", False, ("regression",), _rmse),
        Metric("mae", "MAE", False, ("regression",), lambda y, p: skm.mean_absolute_error(y, p)),
        Metric("r2", "R²", True, ("regression",), lambda y, p: skm.r2_score(y, p)),
        Metric("rmsle", "RMSLE", False, ("regression",), _rmsle),
    ]
}

DEFAULT_METRIC = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}


def resolve_metric(task: str, metric: str | None) -> Metric:
    name = metric or DEFAULT_METRIC[task]
    if name not in METRICS:
        raise ValueError(f"Unknown metric '{name}'. Choose one of: {', '.join(METRICS)}.")
    m = METRICS[name]
    if task not in m.tasks:
        raise ValueError(f"Metric '{name}' does not apply to a {task} task.")
    return m
