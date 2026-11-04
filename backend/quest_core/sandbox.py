"""Run approved feature code in a separate, restricted process.

The code defines `features(df) -> DataFrame` and returns only new columns. Layers of protection:

1. Static check: imports limited to numpy, pandas, math and re; no file, process, network or
   introspection builtins; no underscore attributes; no pandas/numpy I/O helpers.
2. A fresh Python process in isolated mode with an empty environment (no API keys), a temp working
   directory, and CPU, memory, file-size and open-file limits it sets on itself before reading any
   input (no preexec_fn, which is unsafe in threaded parents), plus a wall-clock timeout.
3. Results come back as JSON and .npy files read with allow_pickle=False, so the parent never
   unpickles anything the child wrote.
4. Row-independence check: outputs must not change when rows are processed in different batches.
   Code that uses column means, ranks or row order would leak across CV folds and is rejected.

This is defence in depth for code you have read and approved, not a boundary for hostile code.
For untrusted code, run the worker in a container with no network access.
"""
from __future__ import annotations

import ast
import json
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RUNNER = Path(__file__).with_name("_sandbox_runner.py")
ALLOWED_IMPORTS = {"numpy", "pandas", "math", "re"}
BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "help", "memoryview", "exit", "quit", "dir", "type", "object",
    "classmethod", "staticmethod", "property", "super",
}
BLOCKED_ATTRS = {
    # pandas / numpy I/O and escape hatches
    "read_csv", "read_parquet", "read_pickle", "read_json", "read_excel", "read_sql", "read_html", "read_table",
    "read_feather", "read_hdf", "read_fwf", "read_clipboard", "read_xml", "read_orc", "read_stata", "read_sas",
    "to_csv", "to_parquet", "to_pickle", "to_json", "to_excel", "to_sql", "to_html", "to_feather", "to_hdf",
    "to_clipboard", "to_latex", "to_markdown", "to_xml", "to_orc", "to_stata", "tofile",
    "save", "savez", "savez_compressed", "savetxt", "load", "loadtxt", "genfromtxt", "fromfile", "fromregex",
    "memmap", "ctypes", "ctypeslib", "lib", "io", "testing", "f2py", "distutils", "show_config",
    "eval", "query", "pipe", "options", "set_option", "api",
}
MAX_CODE_CHARS = 20_000
DEFAULT_TIMEOUT = 120
MEMORY_BYTES = 4 * 1024**3


class SandboxError(ValueError):
    pass


def check_code(source: str) -> list[str]:
    """Static checks. Returns human-readable problems; empty means it may run."""
    if not source or not source.strip():
        return ["The code is empty."]
    if len(source) > MAX_CODE_CHARS:
        return [f"Keep feature code under {MAX_CODE_CHARS:,} characters."]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"Line {exc.lineno}: {exc.msg}."]

    problems = []
    if not any(isinstance(n, ast.FunctionDef) and n.name == "features" for n in tree.body):
        problems.append("Define a top-level function `features(df)` that returns a DataFrame of new columns.")
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Expr)):
            problems.append(f"Line {node.lineno}: only imports, constants and functions belong at the top level.")
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"Line {line}: importing '{alias.name}' isn't allowed. Use numpy, pandas, math or re.")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS or node.level:
                problems.append(f"Line {line}: importing from '{node.module}' isn't allowed.")
        elif isinstance(node, ast.Name) and (node.id in BLOCKED_NAMES or node.id.startswith("__")):
            problems.append(f"Line {line}: '{node.id}' isn't available in feature code.")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BLOCKED_ATTRS):
            problems.append(f"Line {line}: '.{node.attr}' isn't available in feature code.")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.Await,
                               ast.Yield, ast.YieldFrom, ast.With, ast.AsyncWith)):
            problems.append(f"Line {line}: '{type(node).__name__}' isn't allowed in feature code.")
    return list(dict.fromkeys(problems))


@dataclass
class SandboxResult:
    columns: dict[str, str]  # name -> "numeric" | "categorical"
    train: pd.DataFrame
    test: pd.DataFrame | None
    seconds: float


def run_features(source: str, train: pd.DataFrame, test: pd.DataFrame | None = None,
                 timeout: int = DEFAULT_TIMEOUT, check_rows: int = 2000) -> SandboxResult:
    problems = check_code(source)
    if problems:
        raise SandboxError(" ".join(problems))

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="quest-sandbox-") as tmp:
        work = Path(tmp)
        with (work / "input.pkl").open("wb") as fh:  # parent -> child only
            pickle.dump({"train": train.reset_index(drop=True), "test": None if test is None else test.reset_index(drop=True),
                         "check_rows": check_rows}, fh)
        (work / "code.py").write_text(source)
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "LANG": "C.UTF-8"}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(RUNNER), tmp, str(timeout), str(MEMORY_BYTES)], cwd=tmp, env=env,
                capture_output=True, text=True, timeout=timeout, start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f"The feature code took longer than {timeout}s and was stopped.") from exc

        error_file = work / "error.json"
        if error_file.exists():
            raise SandboxError(json.loads(error_file.read_text())["message"])
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            reason = "it ran out of memory or CPU time" if proc.returncode < 0 else "it exited unexpectedly"
            raise SandboxError(f"The feature code stopped because {reason}. {' '.join(tail)}".strip())

        meta = json.loads((work / "out" / "meta.json").read_text())
        frames = {}
        for part in meta["parts"]:
            data = {}
            for i, col in enumerate(meta["columns"]):
                if col["kind"] == "numeric":
                    data[col["name"]] = np.load(work / "out" / f"{part}_{i}.npy", allow_pickle=False)
                else:
                    data[col["name"]] = pd.Series(json.loads((work / "out" / f"{part}_{i}.json").read_text()), dtype=object)
            frames[part] = pd.DataFrame(data, index=(train if part == "train" else test).index)
        return SandboxResult(
            columns={c["name"]: c["kind"] for c in meta["columns"]},
            train=frames["train"], test=frames.get("test"), seconds=round(time.perf_counter() - started, 2),
        )
