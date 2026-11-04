"""Leakage and data-integrity checks.

Each finding has a severity:
    high    likely to make CV scores meaningless until resolved
    medium  can distort validation or the public/private gap
    info    worth knowing; no action required
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .univariate import univariate_strength

MAX_UNIVARIATE_FEATURES = 300
_GROUP_NAME_HINT = re.compile(r"(^|_)(id|user|customer|client|patient|subject|session|store|shop|device|account|group|household|site)(_|$)", re.I)


def _finding(severity: str, code: str, title: str, detail: str, columns: list[str] | None = None) -> dict:
    return {"severity": severity, "code": code, "title": title, "detail": detail, "columns": columns or []}


def _is_id_like(series: pd.Series) -> bool:
    """Almost unique per row AND discrete. Continuous floats are unique too, but they aren't IDs."""
    non_null = series.dropna()
    if len(non_null) == 0 or non_null.nunique() / len(non_null) <= 0.98:
        return False
    if pd.api.types.is_numeric_dtype(non_null):
        values = non_null.to_numpy(dtype=float)
        return bool(np.all(values == np.round(values)))
    return True


def _row_hashes(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df.astype(str), index=False)


def check_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    task: str,
    features: list[str],
    kinds: dict[str, str],
    x_codes: pd.DataFrame,
    y: np.ndarray,
    group_col: str | None,
    seed: int = 42,
) -> dict:
    findings: list[dict] = []
    strong_threshold = 0.98 if task != "regression" else 0.95

    # Columns that exist in train only were probably recorded after the outcome.
    reserved = {target}
    train_only = [c for c in train.columns if c not in test.columns and c not in reserved]
    if train_only:
        findings.append(_finding(
            "medium", "train_only_columns", "Columns missing from test",
            "These columns exist in train but not in test, so they can't be features. "
            "If any were recorded after the outcome, they also hint at what drives the target.",
            train_only,
        ))

    constant = [c for c in features if train[c].nunique(dropna=False) <= 1]
    if constant:
        findings.append(_finding("info", "constant_columns", "Constant columns",
                                 "These columns have a single value in train and carry no signal.", constant))

    # Duplicates inside train, and conflicting labels among them.
    feature_frame = train[features]
    hashes = _row_hashes(feature_frame)
    dup_mask = hashes.duplicated(keep=False)
    n_dup = int(hashes.duplicated().sum())
    conflicting = 0
    if n_dup:
        labels = train.loc[dup_mask, target].astype(str)
        conflicting = int((labels.groupby(hashes[dup_mask]).nunique() > 1).sum())
        findings.append(_finding(
            "medium" if n_dup / len(train) > 0.01 else "info", "train_duplicates", "Duplicate rows in train",
            f"{n_dup} rows repeat another row's features exactly"
            + (f"; {conflicting} duplicate groups carry conflicting labels, which caps achievable accuracy." if conflicting else ".")
            + " Random K-fold puts copies on both sides of a split and inflates CV.",
        ))

    # Exact feature overlap between train and test.
    test_hashes = _row_hashes(test[features])
    overlap = int(test_hashes.isin(set(hashes)).sum())
    if overlap:
        share = overlap / len(test)
        findings.append(_finding(
            "medium" if share > 0.01 else "info", "train_test_overlap", "Test rows that also appear in train",
            f"{overlap} test rows ({share:.1%}) match a train row exactly. Their labels are effectively known, "
            "so public-LB gains on these rows won't transfer to the private split.",
        ))

    # Single-feature target proxies, and ID-like columns that predict the target.
    strengths = []
    for col in features[:MAX_UNIVARIATE_FEATURES]:
        if col not in x_codes.columns:
            continue
        s = univariate_strength(x_codes[col].to_numpy(), y, task, seed, categorical=kinds.get(col) == "categorical")
        if s is not None:
            strengths.append({"column": col, "strength": s})
    strengths.sort(key=lambda r: r["strength"], reverse=True)

    proxies = [r["column"] for r in strengths if r["strength"] >= strong_threshold]
    if proxies:
        findings.append(_finding(
            "high", "target_proxy", "Single column nearly predicts the target",
            "On its own, each of these columns predicts the target almost perfectly. That's usually leakage "
            "(a post-outcome field or an encoded label) rather than a real signal. Confirm before using it.",
            proxies,
        ))

    id_like = [c for c in features if kinds.get(c) in ("numeric", "categorical") and _is_id_like(train[c])]
    floor = 0.5 if task != "regression" else 0.0
    predictive_ids = [
        r["column"] for r in strengths
        if r["column"] in id_like and r["strength"] > floor + (0.1 if task != "regression" else 0.05)
    ]
    if predictive_ids:
        findings.append(_finding(
            "high", "predictive_id", "ID-like column predicts the target",
            "These columns are almost unique per row yet carry signal, which usually means the data was "
            "ordered or batched by outcome. Exclude them unless the ordering is genuinely available at test time.",
            predictive_ids,
        ))
    elif id_like:
        findings.append(_finding("info", "id_like_columns", "ID-like columns",
                                 "These columns are unique per row and are still used as features. Drop them unless their values carry real meaning.",
                                 id_like))

    # Row order: if the file order predicts the target, shuffled CV hides a time or batch effect.
    order_strength = univariate_strength(np.arange(len(train), dtype=float), y, task, seed)
    if order_strength is not None and order_strength > floor + (0.15 if task != "regression" else 0.1):
        findings.append(_finding(
            "medium", "row_order_signal", "Row order predicts the target",
            f"The row position alone scores {order_strength:.3f}. The file is likely sorted by time or batch. "
            "If test comes from a later period, use a time-ordered split instead of shuffled K-fold.",
        ))

    # Possible group structure the user hasn't declared.
    group_hints = []
    if not group_col:
        for c in features:
            if not _GROUP_NAME_HINT.search(c):
                continue
            n_unique = train[c].nunique()
            if 2 <= n_unique <= len(train) / 2:
                group_hints.append(c)
    if group_hints:
        findings.append(_finding(
            "medium", "possible_groups", "Possible group column",
            "These columns look like entity identifiers that repeat across rows. If the same entity appears in "
            "train and test, or test holds unseen entities, set one as the group column so CV respects it.",
            group_hints,
        ))

    order = {"high": 0, "medium": 1, "info": 2}
    findings.sort(key=lambda f: order[f["severity"]])
    return {
        "findings": findings,
        "duplicate_rows": n_dup,
        "conflicting_duplicate_groups": conflicting,
        "train_test_overlap_rows": overlap,
        "row_order_strength": order_strength,
        "univariate_strength": strengths[:25],
        "strength_scale": "AUC (0.5 = no signal)" if task != "regression" else "R² (0 = no signal)",
    }
