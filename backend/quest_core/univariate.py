"""How well can a single column predict a target on its own?

Used for leakage (feature vs. label) and for shift (feature vs. "is this row from test?").
A shallow tree with out-of-fold predictions catches non-monotonic relationships that a
plain correlation would miss, while staying cheap enough to run on every column.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MAX_ROWS = 20_000


def _sample(n: int, seed: int) -> np.ndarray:
    if n <= MAX_ROWS:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, MAX_ROWS, replace=False))


def _oof_target_mean(codes: np.ndarray, target: np.ndarray, splitter, split_y, smoothing: float = 10.0) -> np.ndarray:
    """Out-of-fold smoothed target mean per category code (unseen codes get the fold mean)."""
    oof = np.zeros(len(codes), dtype=float)
    for tr, va in splitter.split(codes.reshape(-1, 1), split_y):
        prior = target[tr].mean()
        sums = {}
        for c, t in zip(codes[tr], target[tr]):
            s = sums.setdefault(c, [0.0, 0])
            s[0] += t
            s[1] += 1
        enc = {c: (s + smoothing * prior) / (n + smoothing) for c, (s, n) in sums.items()}
        oof[va] = [enc.get(c, prior) for c in codes[va]]
    return oof


def univariate_strength(
    x: np.ndarray, y: np.ndarray, task: str, seed: int = 42, categorical: bool = False
) -> float | None:
    """Out-of-fold score of one feature on its own.

    Numeric columns use a depth-3 tree; categorical columns use out-of-fold target
    encoding so high-cardinality leaks (an ID that determines the label) are caught.
    Returns AUC (0.5 = useless) for classification, R² (0 = useless) for regression,
    or None when the column carries no usable signal (all missing / constant).
    """
    idx = _sample(len(x), seed)
    x = np.asarray(x, dtype=float)[idx]
    y = np.asarray(y)[idx]
    if np.all(np.isnan(x)) or np.nanstd(x) == 0:
        return None

    if categorical and task != "multiclass":
        codes = np.nan_to_num(x, nan=-1.0)
        if task == "regression":
            splitter = KFold(3, shuffle=True, random_state=seed)
            oof = _oof_target_mean(codes, y.astype(float), splitter, None)
            return float(max(r2_score(y, oof), 0.0))
        classes, y_enc = np.unique(y, return_inverse=True)
        if len(classes) < 2 or np.bincount(y_enc).min() < 3:
            return None
        splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
        oof = _oof_target_mean(codes, y_enc.astype(float), splitter, y_enc)
        return float(max(roc_auc_score(y_enc, oof), 0.5))

    x = x.reshape(-1, 1)

    leaf = max(5, len(y) // 200)
    if task == "regression":
        model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=leaf, random_state=seed)
        splitter = KFold(3, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), dtype=float)
        for tr, va in splitter.split(x):
            oof[va] = model.fit(x[tr], y[tr]).predict(x[va])
        return float(max(r2_score(y, oof), 0.0))

    classes, y_enc = np.unique(y, return_inverse=True)
    counts = np.bincount(y_enc)
    if len(classes) < 2 or counts.min() < 3:
        return None
    model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=leaf, random_state=seed)
    splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), len(classes)), dtype=float)
    for tr, va in splitter.split(x, y_enc):
        fitted = model.fit(x[tr], y_enc[tr])
        oof[np.ix_(va, fitted.classes_)] = fitted.predict_proba(x[va])
    if len(classes) == 2:
        auc = roc_auc_score(y_enc, oof[:, 1])
    else:
        auc = roc_auc_score(y_enc, oof, multi_class="ovr", labels=np.arange(len(classes)))
    return float(max(auc, 0.5))
