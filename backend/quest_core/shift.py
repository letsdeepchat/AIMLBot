"""Train/test distribution shift.

Adversarial validation trains a classifier to tell train rows from test rows. If it can,
the private leaderboard is scored on data that random CV doesn't represent, and the
out-of-fold "test-likeness" of each train row becomes the basis for a test-like holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .univariate import univariate_strength

MAX_ROWS_PER_SIDE = 50_000
MAX_DRIFT_FEATURES = 300


def _band(auc: float) -> str:
    if auc < 0.55:
        return "none"
    if auc < 0.65:
        return "mild"
    if auc < 0.8:
        return "moderate"
    return "severe"


def _classifier(seed: int, cat_mask: list[bool]):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=seed, verbose=-1,
        ), "lightgbm"
    except ImportError:
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, early_stopping=False, random_state=seed,
            categorical_features=cat_mask if any(cat_mask) else None,
        ), "sklearn_hist_gb"


def psi(expected: np.ndarray, actual: np.ndarray, numeric: bool, bins: int = 10) -> float:
    """Population stability index of `actual` against `expected`; missing values get their own bin."""
    eps = 1e-4
    if numeric:
        e_nan, a_nan = np.isnan(expected), np.isnan(actual)
        e_vals, a_vals = expected[~e_nan], actual[~a_nan]
        if len(e_vals) == 0 or len(a_vals) == 0:
            e_counts, a_counts = np.array([]), np.array([])
        else:
            edges = np.unique(np.quantile(e_vals, np.linspace(0, 1, bins + 1)))
            if len(edges) < 2:
                edges = np.array([e_vals.min() - 0.5, e_vals.min() + 0.5])
            edges[0], edges[-1] = -np.inf, np.inf
            e_counts = np.histogram(e_vals, edges)[0]
            a_counts = np.histogram(a_vals, edges)[0]
        e_dist = np.append(e_counts, e_nan.sum()) / len(expected)
        a_dist = np.append(a_counts, a_nan.sum()) / len(actual)
    else:
        e_series, a_series = pd.Series(expected), pd.Series(actual)
        levels = e_series.value_counts().head(50).index
        e_dist = np.append(e_series.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy(),
                           (~e_series.isin(levels)).mean())
        a_dist = np.append(a_series.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy(),
                           (~a_series.isin(levels)).mean())
    e_dist, a_dist = np.clip(e_dist, eps, None), np.clip(a_dist, eps, None)
    return float(np.sum((a_dist - e_dist) * np.log(a_dist / e_dist)))


def _sample_side(n: int, seed: int) -> np.ndarray:
    if n <= MAX_ROWS_PER_SIDE:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, MAX_ROWS_PER_SIDE, replace=False))


def adversarial_validation(
    train_codes: pd.DataFrame, test_codes: pd.DataFrame, cat_cols: list[str], seed: int = 42
) -> tuple[dict, np.ndarray]:
    """Returns the summary and the out-of-fold test-likeness of every train row."""
    tr_idx = _sample_side(len(train_codes), seed)
    te_idx = _sample_side(len(test_codes), seed + 1)
    X = pd.concat([train_codes.iloc[tr_idx], test_codes.iloc[te_idx]], ignore_index=True)
    is_test = np.r_[np.zeros(len(tr_idx)), np.ones(len(te_idx))].astype(int)

    cat_mask = [c in cat_cols and X[c].nunique() < 250 for c in X.columns]
    X_model = X.copy()
    for c, is_cat in zip(X.columns, cat_mask):
        if is_cat:  # HistGradientBoosting wants non-negative codes with NaN for missing
            X_model[c] = X_model[c].where(X_model[c] >= 0)

    oof = np.zeros(len(X))
    importances = np.zeros(X.shape[1])
    engine = None
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X_model, is_test):
        model, engine = _classifier(seed, cat_mask)
        model.fit(X_model.iloc[tr], is_test[tr])
        oof[va] = model.predict_proba(X_model.iloc[va])[:, 1]
        if hasattr(model, "feature_importances_"):
            importances += model.feature_importances_

    auc = float(roc_auc_score(is_test, oof))
    likeness = np.full(len(train_codes), np.nan)
    likeness[tr_idx] = oof[: len(tr_idx)]
    train_like = oof[: len(tr_idx)]

    summary = {
        "auc": auc,
        "band": _band(auc),
        "engine": engine,
        "rows_used": {"train": int(len(tr_idx)), "test": int(len(te_idx))},
        "train_likeness_quantiles": {str(q): float(np.quantile(train_like, q)) for q in (0.5, 0.8, 0.9, 0.95, 0.99)},
        "train_rows_more_test_like_than_not": int((train_like > 0.5).sum()),
    }
    if importances.any():
        order = np.argsort(importances)[::-1][:15]
        total = importances.sum()
        summary["model_importance"] = [
            {"column": X.columns[i], "share": float(importances[i] / total)} for i in order if importances[i] > 0
        ]
    return summary, likeness


def feature_drift(
    train_prepared: pd.DataFrame, test_prepared: pd.DataFrame,
    train_codes: pd.DataFrame, test_codes: pd.DataFrame, seed: int = 42,
) -> list[dict]:
    rows = []
    n_tr, n_te = len(train_codes), len(test_codes)
    for col in list(train_codes.columns)[:MAX_DRIFT_FEATURES]:
        numeric = train_prepared[col].dtype != object
        tr_raw = train_prepared[col].to_numpy()
        te_raw = test_prepared[col].to_numpy()
        row = {"column": col, "kind": "numeric" if numeric else "categorical"}
        row["psi"] = psi(tr_raw.astype(float) if numeric else tr_raw, te_raw.astype(float) if numeric else te_raw, numeric)
        if numeric:
            a, b = tr_raw[~np.isnan(tr_raw)], te_raw[~np.isnan(te_raw)]
            if len(a) and len(b):
                ks = stats.ks_2samp(a, b)
                row["ks_stat"], row["ks_pvalue"] = float(ks.statistic), float(ks.pvalue)
        else:
            row["unseen_in_test"] = float((~pd.Series(te_raw).isin(set(tr_raw))).mean())
        x = np.r_[train_codes[col].to_numpy(), test_codes[col].to_numpy()]
        is_test = np.r_[np.zeros(n_tr), np.ones(n_te)]
        row["adversarial_auc"] = univariate_strength(x, is_test, "binary", seed, categorical=not numeric)
        rows.append(row)
    rows.sort(key=lambda r: (r["adversarial_auc"] or 0.5, r["psi"]), reverse=True)
    return rows


def analyze_shift(
    train_prepared: pd.DataFrame, test_prepared: pd.DataFrame,
    train_codes: pd.DataFrame, test_codes: pd.DataFrame,
    cat_cols: list[str], seed: int = 42,
    time_values: tuple[pd.Series, pd.Series] | None = None,
) -> tuple[dict, np.ndarray]:
    summary, likeness = adversarial_validation(train_codes, test_codes, cat_cols, seed)
    drift = feature_drift(train_prepared, test_prepared, train_codes, test_codes, seed)
    result = {
        "adversarial": summary,
        "features": drift,
        "drifting_features": [r["column"] for r in drift if r["psi"] > 0.25 or (r["adversarial_auc"] or 0.5) > 0.7],
    }
    if time_values is not None:
        tr_t, te_t = time_values
        if tr_t.notna().any() and te_t.notna().any():
            result["time"] = {
                "train_range": [float(tr_t.min()), float(tr_t.max())],
                "test_range": [float(te_t.min()), float(te_t.max())],
                "test_after_train": bool(te_t.min() >= tr_t.max()),
                "overlap_share": float(((te_t >= tr_t.min()) & (te_t <= tr_t.max())).mean()),
            }
    return result, likeness
