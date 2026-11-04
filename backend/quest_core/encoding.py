"""Column typing and feature preparation shared by every analysis module.

Kinds:
    numeric      numbers and booleans, used as float
    datetime     strings that parse as dates, used as days since epoch
    categorical  everything else with modest text length
    text         long free text, excluded from week-1 baselines
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

MISSING = "__missing__"
_DATE_PATTERN = re.compile(r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})")


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().astype(str).head(200)
    if sample.empty or sample.str.match(_DATE_PATTERN).mean() < 0.9:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.9


def detect_kind(series: pd.Series) -> str:
    if is_numeric(series):
        return "numeric"
    if _looks_like_datetime(series):
        return "datetime"
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return "categorical"
    avg_len = non_null.str.len().mean()
    unique_ratio = non_null.nunique() / len(non_null)
    if avg_len > 40 and unique_ratio > 0.3:
        return "text"
    return "categorical"


def column_kinds(train: pd.DataFrame, features: list[str]) -> dict[str, str]:
    return {col: detect_kind(train[col]) for col in features}


def _to_epoch_days(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(None)
    return (parsed - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)


def prepare_features(df: pd.DataFrame, kinds: dict[str, str]) -> pd.DataFrame:
    """Return a model-ready frame: floats for numeric/datetime, clean strings for categoricals.

    Text columns are dropped. Categorical missing values become an explicit level so
    every downstream encoder sees plain Python strings (safe with pandas 3 string dtype).
    """
    out = {}
    for col, kind in kinds.items():
        s = df[col]
        if kind == "numeric":
            out[col] = pd.to_numeric(s, errors="coerce").astype(float)
        elif kind == "datetime":
            out[col] = _to_epoch_days(s).astype(float)
        elif kind == "categorical":
            values = s.astype(object).where(s.notna(), MISSING)
            out[col] = values.map(str).astype(object)
    return pd.DataFrame(out, index=df.index)


def ordinal_codes(train: pd.DataFrame, test: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Encode categoricals as integer codes shared across train and test (NaN stays NaN)."""
    train = train.copy()
    test = test.copy() if test is not None else None
    for col in train.columns:
        if train[col].dtype == object:
            combined = train[col] if test is None else pd.concat([train[col], test[col]])
            categories = pd.Index(combined.unique())
            train[col] = categories.get_indexer(train[col]).astype(float)
            if test is not None:
                test[col] = categories.get_indexer(test[col]).astype(float)
    return train, test


def categorical_columns(prepared: pd.DataFrame) -> list[str]:
    return [c for c in prepared.columns if prepared[c].dtype == object]


def numeric_columns(prepared: pd.DataFrame) -> list[str]:
    return [c for c in prepared.columns if prepared[c].dtype != object]


def jsonable(value):
    """Recursively convert numpy/pandas scalars so the report can be written with json.dump."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if np.isnan(f) or np.isinf(f) else f
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if value is pd.NA or value is pd.NaT:
        return None
    return value
