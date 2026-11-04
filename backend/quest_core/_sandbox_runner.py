"""Child process for sandbox.run_features. Started with `python -I`, an empty environment and rlimits.

Reads input.pkl (written by the parent), executes code.py with restricted builtins, and writes results
as JSON / .npy so the parent never has to unpickle anything produced here.
"""
import builtins
import json
import math
import pickle
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(sys.argv[1])


def _limit_self() -> None:
    """Lower this process's own limits before any input or user code is touched. Can't be raised again."""
    try:
        import resource
    except ImportError:  # not POSIX
        return
    timeout, memory = int(sys.argv[2]), int(sys.argv[3])
    for res, value in ((resource.RLIMIT_CPU, timeout + 5), (resource.RLIMIT_AS, memory),
                       (resource.RLIMIT_FSIZE, 256 * 1024**2), (resource.RLIMIT_NOFILE, 256), (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(res, (value, value))


_limit_self()
MISSING = "__missing__"
ALLOWED = {"numpy", "pandas", "math", "re"}
SAFE = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "frozenset", "int", "isinstance", "len",
    "list", "map", "max", "min", "pow", "range", "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "zip", "divmod", "hash", "iter", "next", "callable", "chr", "ord", "repr", "format",
    "ValueError", "TypeError", "KeyError", "IndexError", "ZeroDivisionError", "Exception", "ArithmeticError",
    "True", "False", "None", "NotImplemented",
]


def fail(message: str) -> None:
    (WORK / "error.json").write_text(json.dumps({"message": message[:4000]}))
    sys.exit(3)


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED:
        raise ImportError(f"Importing '{name}' isn't allowed in feature code.")
    return builtins.__import__(name, globals, locals, fromlist, level)


def short_trace() -> str:
    lines = traceback.format_exc().strip().splitlines()
    user = [ln for ln in lines if "features.py" in ln or not ln.startswith("  ")]
    return "\n".join(user[-6:])


def main() -> None:
    with (WORK / "input.pkl").open("rb") as fh:
        payload = pickle.load(fh)
    train, test, check_rows = payload["train"], payload["test"], payload["check_rows"]

    safe_builtins = {n: getattr(builtins, n) for n in SAFE if hasattr(builtins, n)}
    safe_builtins["__import__"] = guarded_import
    namespace = {"__builtins__": safe_builtins, "__name__": "features", "np": np, "pd": pd, "math": math, "re": re}
    try:
        exec(compile((WORK / "code.py").read_text(), "features.py", "exec"), namespace)
    except Exception:
        fail(f"The code failed to load:\n{short_trace()}")
    fn = namespace.get("features")
    if not callable(fn):
        fail("Define a function `features(df)`.")

    def call(df: pd.DataFrame) -> pd.DataFrame:
        try:
            out = fn(df.copy())
        except Exception:
            fail(f"features(df) raised an error:\n{short_trace()}")
        if isinstance(out, pd.Series):
            out = out.to_frame()
        if not isinstance(out, pd.DataFrame):
            fail("features(df) must return a pandas DataFrame of new columns.")
        if len(out) != len(df) or not out.index.equals(df.index):
            fail("features(df) must return one row per input row, with the input's index. Build columns from df directly.")
        clash = [c for c in out.columns if c in df.columns]
        if clash:
            fail(f"Return only new columns; these already exist: {', '.join(map(str, clash[:5]))}.")
        if not len(out.columns):
            fail("features(df) returned no columns.")
        out.columns = [str(c) for c in out.columns]
        return out

    full = call(train)

    # Row independence: batching must not change any value, or the code depends on other rows.
    if len(train) >= 6:
        sample = train.sample(min(len(train), check_rows), random_state=0)
        whole = call(sample)
        chunks = np.array_split(np.random.default_rng(1).permutation(len(sample)), 3)
        pieced = pd.concat([call(sample.iloc[np.sort(c)]) for c in chunks]).loc[sample.index]
        for col in whole.columns:
            a, b = whole[col], pieced[col]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                same = np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=1e-9, atol=1e-12, equal_nan=True)
            else:
                same = a.astype(str).fillna(MISSING).equals(b.astype(str).fillna(MISSING))
            if not same:
                fail(
                    f"Column '{col}' changes when rows are processed in different batches, so it depends on other "
                    "rows (a column mean, rank, shift or similar). That would leak across CV folds. Keep feature code "
                    "row-by-row, and use recipe steps such as group statistics, clip or target encoding for anything "
                    "learned from the data, since those are fitted per fold."
                )

    parts = {"train": full}
    if test is not None:
        parts["test"] = call(test)

    out_dir = WORK / "out"
    out_dir.mkdir()
    columns = []
    for i, col in enumerate(full.columns):
        series = full[col]
        numeric = (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series))
        columns.append({"name": col, "kind": "numeric" if numeric else "categorical"})
        for part, frame in parts.items():
            values = frame[col]
            if numeric:
                arr = np.array(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float), copy=True)
                arr[~np.isfinite(arr)] = np.nan
                np.save(out_dir / f"{part}_{i}.npy", arr, allow_pickle=False)
            else:
                texts = [MISSING if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v) for v in values]
                (out_dir / f"{part}_{i}.json").write_text(json.dumps(texts))
    (out_dir / "meta.json").write_text(json.dumps({"columns": columns, "parts": list(parts)}))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        fail(f"The sandbox couldn't process the output:\n{short_trace()}")
