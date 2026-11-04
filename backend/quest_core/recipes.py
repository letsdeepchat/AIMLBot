"""Feature recipes: ordered, declarative feature steps that are safe to cross-validate.

A recipe is an sklearn transformer placed first in the model pipeline, so every step that learns
from data (target encoding, group statistics, frequencies, clip thresholds) learns from the
training fold only. Target encoding additionally cross-fits inside the fold, so training rows
never see an encoding computed from their own label.

Formulas are parsed into an AST and evaluated against a whitelist of numpy functions. They can
combine numeric columns; they cannot run arbitrary code.
"""
from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

NUMERIC_KINDS = ("numeric", "datetime")

# --------------------------------------------------------------------------- catalog
# Drives both validation and the UI's step builder. Param kinds:
#   text, expression, number, column, columns, options

STEP_CATALOG: list[dict] = [
    {"type": "formula", "label": "Formula",
     "help": "A new numeric column from an expression, e.g. log1p(income) / (age + 1). Use col(\"name\") for names with spaces.",
     "params": [{"name": "name", "kind": "text", "label": "New column"},
                {"name": "expression", "kind": "expression", "label": "Expression"}]},
    {"type": "log1p", "label": "Log transform",
     "help": "Replaces each column with sign(x) · log1p(|x|). Tames heavy right tails.",
     "params": [{"name": "columns", "kind": "columns", "accepts": ["numeric"], "label": "Columns"}]},
    {"type": "clip", "label": "Clip outliers",
     "help": "Clips each column to quantiles learned on the training fold.",
     "params": [{"name": "columns", "kind": "columns", "accepts": ["numeric"], "label": "Columns"},
                {"name": "lower", "kind": "number", "label": "Lower quantile", "default": 0.01, "min": 0, "max": 0.5},
                {"name": "upper", "kind": "number", "label": "Upper quantile", "default": 0.99, "min": 0.5, "max": 1}]},
    {"type": "missing_flags", "label": "Missing indicators",
     "help": "Adds a 0/1 column marking where each column is missing.",
     "params": [{"name": "columns", "kind": "columns", "accepts": ["numeric", "categorical", "datetime"], "label": "Columns"}]},
    {"type": "row_stats", "label": "Row statistics",
     "help": "Mean, std, min, max and missing count across the chosen columns, per row.",
     "params": [{"name": "columns", "kind": "columns", "accepts": ["numeric"], "label": "Columns"},
                {"name": "prefix", "kind": "text", "label": "Name prefix", "default": "row"}]},
    {"type": "datetime_parts", "label": "Date parts",
     "help": "Year, month, day of week, day of year and hour from each date column.",
     "params": [{"name": "columns", "kind": "columns", "accepts": ["datetime"], "label": "Date columns"}]},
    {"type": "frequency_encode", "label": "Frequency encoding",
     "help": "Each category's share of the training fold. Unseen categories get 0.",
     "params": [{"name": "columns", "kind": "columns", "accepts": ["categorical"], "label": "Columns"}]},
    {"type": "target_encode", "label": "Target encoding",
     "help": "Smoothed target mean per category, cross-fitted inside each training fold so it can't leak.",
     "params": [{"name": "columns", "kind": "columns", "accepts": ["categorical"], "label": "Columns"},
                {"name": "smoothing", "kind": "number", "label": "Smoothing (rows)", "default": 20, "min": 0, "max": 10000}]},
    {"type": "group_agg", "label": "Group statistics",
     "help": "Statistics of numeric columns within each group, learned on the training fold.",
     "params": [{"name": "by", "kind": "column", "accepts": ["categorical"], "label": "Group by"},
                {"name": "columns", "kind": "columns", "accepts": ["numeric"], "label": "Columns"},
                {"name": "stats", "kind": "options", "label": "Statistics",
                 "options": ["mean", "std", "min", "max", "count", "diff_from_mean"], "default": ["mean", "diff_from_mean"]}]},
    {"type": "combine", "label": "Combine categories",
     "help": "A new categorical column from two categoricals, e.g. city × plan.",
     "params": [{"name": "first", "kind": "column", "accepts": ["categorical"], "label": "First"},
                {"name": "second", "kind": "column", "accepts": ["categorical"], "label": "Second"}]},
    {"type": "drop", "label": "Drop columns",
     "help": "Removes columns from the model input.",
     "params": [{"name": "columns", "kind": "columns", "accepts": ["numeric", "categorical", "datetime"], "label": "Columns"}]},
]
CATALOG = {s["type"]: s for s in STEP_CATALOG}
GROUP_STATS = ("mean", "std", "min", "max", "count", "diff_from_mean")

# --------------------------------------------------------------------------- formulas

_FUNCS = {
    "log": np.log, "log1p": np.log1p, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
    "floor": np.floor, "round": np.round, "minimum": np.minimum, "maximum": np.maximum,
    "where": np.where, "clip": np.clip,
}
_BINOPS = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
    ast.Pow: np.power, ast.Mod: np.mod, ast.FloorDiv: np.floor_divide,
    ast.BitAnd: np.logical_and, ast.BitOr: np.logical_or,
}
_CMPOPS = {ast.Gt: np.greater, ast.GtE: np.greater_equal, ast.Lt: np.less, ast.LtE: np.less_equal,
           ast.Eq: np.equal, ast.NotEq: np.not_equal}
MAX_FORMULA_CHARS = 500
MAX_FORMULA_NODES = 200


class FormulaError(ValueError):
    pass


def parse_formula(expression: str) -> tuple[ast.Expression, list[str]]:
    """Parse and validate a formula. Returns the tree and the column names it references."""
    if not expression or not expression.strip():
        raise FormulaError("The expression is empty.")
    if len(expression) > MAX_FORMULA_CHARS:
        raise FormulaError(f"Keep expressions under {MAX_FORMULA_CHARS} characters.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"Can't parse the expression: {exc.msg}.") from exc

    refs: list = []
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_FORMULA_NODES:
        raise FormulaError("The expression is too long; split it into two formula steps.")
    for node in nodes:
        if isinstance(node, (ast.Expression, ast.Load, ast.operator, ast.unaryop, ast.cmpop)):
            continue
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BINOPS:
                raise FormulaError(f"The operator {type(node.op).__name__} isn't supported.")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
                raise FormulaError("Only +, - and ~ work as unary operators.")
        elif isinstance(node, ast.Compare):
            if not all(type(op) in _CMPOPS for op in node.ops):
                raise FormulaError("Only <, <=, >, >=, == and != comparisons are supported.")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, str)) or isinstance(node.value, bool):
                raise FormulaError("Only numbers (and column names inside col()) can be constants.")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise FormulaError("Call functions by name with positional arguments only.")
            name = node.func.id
            if name == "col":
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    raise FormulaError('col() takes one quoted column name, like col("monthly spend").')
                refs.append((node.col_offset, node.args[0].value))
            elif name not in _FUNCS:
                raise FormulaError(f"'{name}' isn't an allowed function. Use: {', '.join(sorted(_FUNCS))}.")
        elif isinstance(node, ast.Name):
            continue  # function names are checked on the Call; the rest are columns (collected below)
        else:
            raise FormulaError(f"'{type(node).__name__}' isn't allowed in an expression.")

    callees = {id(n.func) for n in nodes if isinstance(n, ast.Call)}
    refs += [(n.col_offset, n.id) for n in nodes if isinstance(n, ast.Name) and id(n) not in callees]
    refs = [name for _, name in sorted(refs)]
    for node in nodes:  # strings are only legal as the argument of col()
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            parent_ok = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "col"
                            and c.args and c.args[0] is node for c in nodes)
            if not parent_ok:
                raise FormulaError("Text is only allowed as the column name inside col().")
    return tree, list(dict.fromkeys(refs))


def eval_formula(tree: ast.Expression, frame: pd.DataFrame) -> np.ndarray:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return frame[node.id].to_numpy(dtype=float)
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            value = ev(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.Invert):
                return np.logical_not(value)
            return value
        if isinstance(node, ast.Compare):
            left, result = ev(node.left), None
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                part = _CMPOPS[type(op)](left, right)
                result = part if result is None else np.logical_and(result, part)
                left = right
            return result
        if isinstance(node, ast.Call):
            if node.func.id == "col":
                return frame[node.args[0].value].to_numpy(dtype=float)
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise FormulaError(f"Unsupported node {type(node).__name__}.")

    with np.errstate(all="ignore"):
        out = np.asarray(ev(tree), dtype=float)
    if out.ndim == 0:
        out = np.full(len(frame), float(out))
    out[~np.isfinite(out)] = np.nan  # division by zero, log of negatives
    return out


# --------------------------------------------------------------------------- validation

_NAME = re.compile(r"^[^\s].{0,79}$")


def validate_recipe(steps: list[dict], kinds: dict[str, str], task: str = "binary", n_classes: int = 2) -> dict:
    """Check every step against the columns available at that point, without touching data.

    Returns {"errors": [{"step": i, "message": ...}], "columns_before": [[{name, kind}], ...], "output": [...]}.
    The UI uses columns_before to offer only columns that exist when each step runs.
    """
    errors: list[dict] = []
    available = dict(kinds)
    columns_before = []

    def err(i, message):
        errors.append({"step": i, "message": message})

    for i, step in enumerate(steps):
        columns_before.append([{"name": c, "kind": k} for c, k in available.items()])
        meta = CATALOG.get(step.get("type"))
        if not meta:
            err(i, f"Unknown step type '{step.get('type')}'.")
            continue
        params = step.get("params") or {}

        def need(name, accepts):
            col = params.get(name)
            if not col:
                label = next((p["label"] for p in meta["params"] if p["name"] == name), name)
                err(i, f"Choose a column for {label}.")
                return None
            if col not in available:
                err(i, f"'{col}' doesn't exist at this step.")
                return None
            if available[col] not in accepts:
                err(i, f"'{col}' is {available[col]}; this needs {' or '.join(accepts)}.")
                return None
            return col

        def need_many(accepts):
            cols = params.get("columns") or []
            if not cols:
                err(i, "Choose at least one column.")
            good = []
            for c in cols:
                if c not in available:
                    err(i, f"'{c}' doesn't exist at this step.")
                elif available[c] not in accepts:
                    err(i, f"'{c}' is {available[c]}; this step needs {' or '.join(accepts)}.")
                else:
                    good.append(c)
            return good

        t = step["type"]
        if t == "formula":
            name = (params.get("name") or "").strip()
            if not name or not _NAME.match(name):
                err(i, "Give the new column a name (up to 80 characters).")
            try:
                _, refs = parse_formula(params.get("expression", ""))
                for r in refs:
                    if r not in available:
                        err(i, f"'{r}' doesn't exist at this step. Wrap names with spaces in col(\"...\").")
                    elif available[r] not in NUMERIC_KINDS:
                        err(i, f"'{r}' is {available[r]}; formulas work on numeric columns.")
            except FormulaError as exc:
                err(i, str(exc))
            if name:
                available[name] = "numeric"
        elif t in ("log1p", "clip"):
            need_many(["numeric"])
            if t == "clip":
                lo, hi = float(params.get("lower", 0.01)), float(params.get("upper", 0.99))
                if not 0 <= lo < hi <= 1:
                    err(i, "Quantiles must satisfy 0 ≤ lower < upper ≤ 1.")
        elif t == "missing_flags":
            for c in need_many(["numeric", "categorical", "datetime"]):
                available[f"{c}__missing"] = "numeric"
        elif t == "row_stats":
            cols = need_many(["numeric", "datetime"])
            if len(cols) < 2:
                err(i, "Row statistics need at least two columns.")
            prefix = (params.get("prefix") or "row").strip() or "row"
            for s in ("mean", "std", "min", "max", "n_missing"):
                available[f"{prefix}__{s}"] = "numeric"
        elif t == "datetime_parts":
            for c in need_many(["datetime"]):
                for part in ("year", "month", "dow", "doy", "hour"):
                    available[f"{c}__{part}"] = "numeric"
        elif t == "frequency_encode":
            for c in need_many(["categorical"]):
                available[f"{c}__freq"] = "numeric"
        elif t == "target_encode":
            for c in need_many(["categorical"]):
                if task == "multiclass":
                    for k in range(n_classes):
                        available[f"{c}__te_{k}"] = "numeric"
                else:
                    available[f"{c}__te"] = "numeric"
            if float(params.get("smoothing", 20)) < 0:
                err(i, "Smoothing can't be negative.")
        elif t == "group_agg":
            by = need("by", ["categorical"])
            cols = need_many(["numeric", "datetime"])
            stats = params.get("stats") or []
            bad = [s for s in stats if s not in GROUP_STATS]
            if not stats or bad:
                err(i, f"Choose statistics from: {', '.join(GROUP_STATS)}.")
            if by:
                if "count" in stats:
                    available[f"{by}__count"] = "numeric"
                for c in cols:
                    for s in stats:
                        if s != "count":
                            available[f"{c}__by_{by}__{s}"] = "numeric"
        elif t == "combine":
            a, b = need("first", ["categorical"]), need("second", ["categorical"])
            if a and b:
                if a == b:
                    err(i, "Pick two different columns.")
                available[f"{a}__x__{b}"] = "categorical"
        elif t == "drop":
            for c in need_many(["numeric", "categorical", "datetime"]):
                available.pop(c, None)

    if not available:
        errors.append({"step": len(steps) - 1, "message": "The recipe drops every column."})
    return {"errors": errors, "columns_before": columns_before,
            "output": [{"name": c, "kind": k} for c, k in available.items()]}


# --------------------------------------------------------------------------- transformer


def _target_matrix(y: np.ndarray, task: str, n_classes: int) -> np.ndarray:
    if task == "multiclass":
        return np.eye(n_classes)[y.astype(int)]
    return np.asarray(y, dtype=float).reshape(-1, 1)


def _te_table(keys: pd.Series, target: np.ndarray, smoothing: float) -> tuple[pd.DataFrame, np.ndarray]:
    prior = target.mean(axis=0)
    frame = pd.DataFrame(target, index=keys.index)
    grouped = frame.groupby(keys.to_numpy())
    sums, counts = grouped.sum(), grouped.size()
    table = (sums.add(smoothing * prior, axis=1)).div(counts + smoothing, axis=0)
    return table, prior


def _te_apply(keys: pd.Series, table: pd.DataFrame, prior: np.ndarray) -> np.ndarray:
    out = np.array(table.reindex(keys.to_numpy()).to_numpy(dtype=float), copy=True)
    missing = np.isnan(out).any(axis=1)
    out[missing] = prior
    return out


class FeatureRecipe(BaseEstimator, TransformerMixin):
    """Apply recipe steps; learned state comes only from the data passed to fit."""

    def __init__(self, steps=None, task="binary", seed=42):
        self.steps = steps
        self.task = task
        self.seed = seed

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None, **_):
        self.n_classes_ = int(np.max(y)) + 1 if (y is not None and self.task == "multiclass") else 1
        self.state_ = []
        out = X.copy()
        for step in self.steps or []:
            out, state = self._apply(step, out, y, fit=True, state=None)
            self.state_.append(state)
        self.columns_ = list(out.columns)
        return out

    def transform(self, X):
        out = X.copy()
        for step, state in zip(self.steps or [], self.state_):
            out, _ = self._apply(step, out, None, fit=False, state=state)
        return out[self.columns_]

    # Each branch returns (frame, state). On fit, state is learned; on transform, it's reused.
    def _apply(self, step, out, y, fit, state):
        t, p = step["type"], step.get("params") or {}
        out = out.copy()

        if t == "formula":
            tree, _ = parse_formula(p["expression"])
            out[p["name"].strip()] = eval_formula(tree, out)
            return out, None

        if t == "log1p":
            for c in p["columns"]:
                v = out[c].to_numpy(dtype=float)
                out[c] = np.sign(v) * np.log1p(np.abs(v))
            return out, None

        if t == "clip":
            if fit:
                state = {c: (out[c].quantile(float(p.get("lower", 0.01))), out[c].quantile(float(p.get("upper", 0.99))))
                         for c in p["columns"]}
            for c, (lo, hi) in state.items():
                out[c] = out[c].clip(lo, hi)
            return out, state

        if t == "missing_flags":
            for c in p["columns"]:
                out[f"{c}__missing"] = out[c].isna().astype(float) if out[c].dtype != object else \
                    (out[c] == "__missing__").astype(float)
            return out, None

        if t == "row_stats":
            prefix = (p.get("prefix") or "row").strip() or "row"
            block = out[p["columns"]].astype(float)
            with np.errstate(all="ignore"):
                out[f"{prefix}__mean"] = block.mean(axis=1)
                out[f"{prefix}__std"] = block.std(axis=1)
                out[f"{prefix}__min"] = block.min(axis=1)
                out[f"{prefix}__max"] = block.max(axis=1)
                out[f"{prefix}__n_missing"] = block.isna().sum(axis=1).astype(float)
            return out, None

        if t == "datetime_parts":
            for c in p["columns"]:
                ts = pd.to_datetime(out[c], unit="D", origin="unix", errors="coerce")
                out[f"{c}__year"] = ts.dt.year.astype(float)
                out[f"{c}__month"] = ts.dt.month.astype(float)
                out[f"{c}__dow"] = ts.dt.dayofweek.astype(float)
                out[f"{c}__doy"] = ts.dt.dayofyear.astype(float)
                out[f"{c}__hour"] = ts.dt.hour.astype(float)
            return out, None

        if t == "frequency_encode":
            if fit:
                state = {c: out[c].value_counts(normalize=True) for c in p["columns"]}
            for c, freq in state.items():
                out[f"{c}__freq"] = out[c].map(freq).astype(float).fillna(0.0)
            return out, state

        if t == "target_encode":
            smoothing = float(p.get("smoothing", 20))
            if fit:
                if y is None:
                    raise ValueError("Target encoding needs the target during fit.")
                target = _target_matrix(np.asarray(y), self.task, self.n_classes_)
                state = {}
                for c in p["columns"]:
                    keys = out[c].astype(str)
                    # Cross-fit: each training row is encoded by a table built without it.
                    encoded = np.zeros(target.shape)
                    folds = KFold(5, shuffle=True, random_state=self.seed) if len(keys) >= 10 else None
                    if folds:
                        for tr, va in folds.split(keys):
                            table, prior = _te_table(keys.iloc[tr], target[tr], smoothing)
                            encoded[va] = _te_apply(keys.iloc[va], table, prior)
                    full_table, full_prior = _te_table(keys, target, smoothing)
                    if not folds:
                        encoded = _te_apply(keys, full_table, full_prior)
                    state[c] = (full_table, full_prior)
                    self._write_te(out, c, encoded)
                return out, state
            for c, (table, prior) in state.items():
                self._write_te(out, c, _te_apply(out[c].astype(str), table, prior))
            return out, state

        if t == "group_agg":
            by, cols, stats = p["by"], p["columns"], p.get("stats") or ["mean"]
            if fit:
                state = {"size": out.groupby(by).size()}
                agg = [s for s in stats if s in ("mean", "std", "min", "max")]
                if "diff_from_mean" in stats and "mean" not in agg:
                    agg.append("mean")
                state["agg"] = out.groupby(by)[cols].agg(agg) if agg else None
            keys = out[by]
            if "count" in stats:
                out[f"{by}__count"] = keys.map(state["size"]).astype(float).fillna(0.0)
            for c in cols:
                for s in stats:
                    if s == "count":
                        continue
                    mean = keys.map(state["agg"][(c, "mean")]).astype(float) if s == "diff_from_mean" else None
                    if s == "diff_from_mean":
                        out[f"{c}__by_{by}__{s}"] = out[c].astype(float) - mean
                    else:
                        out[f"{c}__by_{by}__{s}"] = keys.map(state["agg"][(c, s)]).astype(float)
            return out, state

        if t == "combine":
            a, b = p["first"], p["second"]
            out[f"{a}__x__{b}"] = (out[a].astype(str) + "|" + out[b].astype(str)).astype(object)
            return out, None

        if t == "drop":
            return out.drop(columns=[c for c in p["columns"] if c in out.columns]), None

        raise ValueError(f"Unknown step type '{t}'.")

    def _write_te(self, out, column, encoded):
        if encoded.shape[1] == 1:
            out[f"{column}__te"] = encoded[:, 0]
        else:  # multiclass: one probability per class
            for k in range(encoded.shape[1]):
                out[f"{column}__te_{k}"] = encoded[:, k]
