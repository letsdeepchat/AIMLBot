"""Dataset, column and target profiling."""
from __future__ import annotations

import pandas as pd


def size_class(n: int) -> str:
    if n < 1_000:
        return "tiny"
    if n < 10_000:
        return "small"
    if n < 100_000:
        return "medium"
    return "large"


def _column_profile(name: str, kind: str, train: pd.Series, test: pd.Series | None) -> dict:
    non_null = train.dropna()
    n_unique = int(non_null.nunique())
    profile = {
        "name": name,
        "kind": kind,
        "dtype": str(train.dtype),
        "missing_train": float(train.isna().mean()),
        "missing_test": float(test.isna().mean()) if test is not None else None,
        "n_unique": n_unique,
        "unique_ratio": n_unique / max(len(non_null), 1),
        "constant": n_unique <= 1,
    }
    if kind == "numeric" and len(non_null):
        values = pd.to_numeric(non_null, errors="coerce").astype(float)
        profile.update(
            mean=float(values.mean()),
            std=float(values.std()) if len(values) > 1 else 0.0,
            min=float(values.min()),
            median=float(values.median()),
            max=float(values.max()),
            skew=float(values.skew()) if len(values) > 2 else 0.0,
        )
    elif kind in ("categorical", "text", "datetime"):
        top = non_null.astype(str).value_counts().head(5)
        profile["top_values"] = [{"value": v, "count": int(c)} for v, c in top.items()]
        if test is not None and kind == "categorical":
            train_levels = set(non_null.astype(str))
            test_values = test.dropna().astype(str)
            profile["unseen_in_test"] = float((~test_values.isin(train_levels)).mean()) if len(test_values) else 0.0
    return profile


def target_profile(y: pd.Series, task: str) -> dict:
    out = {"name": y.name, "task": task, "missing": int(y.isna().sum())}
    y = y.dropna()
    if task in ("binary", "multiclass"):
        counts = y.astype(str).value_counts()
        out["classes"] = [{"label": k, "count": int(v), "share": float(v / len(y))} for k, v in counts.items()]
        out["imbalance_ratio"] = float(counts.max() / counts.min())
        out["min_class_count"] = int(counts.min())
    else:
        values = y.astype(float)
        skew = float(values.skew()) if len(values) > 2 else 0.0
        out.update(
            mean=float(values.mean()),
            std=float(values.std()),
            min=float(values.min()),
            max=float(values.max()),
            skew=skew,
            quantiles={str(q): float(values.quantile(q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)},
            suggest_log_target=bool(skew > 2 and values.min() >= 0),
        )
    return out


def profile_dataset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    task: str,
    kinds: dict[str, str],
) -> dict:
    columns = [
        _column_profile(col, kind, train[col], test[col] if col in test.columns else None)
        for col, kind in kinds.items()
    ]
    kind_counts = pd.Series(list(kinds.values())).value_counts().to_dict() if kinds else {}
    return {
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(kinds),
        "size_class": size_class(len(train)),
        "kind_counts": {k: int(v) for k, v in kind_counts.items()},
        "memory_mb": float((train.memory_usage(deep=True).sum() + test.memory_usage(deep=True).sum()) / 1e6),
        "target": target_profile(train[target], task),
        "columns": columns,
        "high_missing": [c["name"] for c in columns if c["missing_train"] > 0.5],
        "excluded_text": [c["name"] for c in columns if c["kind"] == "text"],
    }
