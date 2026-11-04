"""Week-2 tests: formula safety, recipe validation, fold-safe encodings, experiments and the leaderboard."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from quest_core import ChallengeSpec, build_leaderboard, run_analysis, run_baselines, run_experiment
from quest_core.experiments import create_experiment, list_experiments
from quest_core.recipes import FeatureRecipe, FormulaError, eval_formula, parse_formula, validate_recipe


def _ratio_data(seed=0, n=500, n_test=200):
    """Target is "a and b are close": non-monotonic in a and b, so a linear model on them can't see it."""
    rng = np.random.default_rng(seed)

    def frame(m):
        a, b = rng.uniform(1, 10, m), rng.uniform(1, 10, m)
        noise = rng.normal(0, 0.05, m)
        df = pd.DataFrame({
            "row_id": np.arange(m), "a": a, "b": b,
            "shop": rng.choice(list("pqrstu"), m), "junk": rng.normal(size=m),
            "noise_id": [f"n{v}" for v in rng.integers(0, m * 2, m)],  # near-unique, carries no signal
        })
        df["target"] = (np.abs(np.log(a / b)) + noise < 0.25).astype(int)
        return df

    train, test = frame(n), frame(n_test).drop(columns="target")
    test["row_id"] += n
    return train, test


def _setup(tmp: Path, **spec_kw):
    train, test = _ratio_data()
    train.to_csv(tmp / "train.csv", index=False)
    test.to_csv(tmp / "test.csv", index=False)
    spec = ChallengeSpec(target="target", id_col="row_id", **spec_kw)
    run_analysis(tmp, spec)
    run_baselines(tmp, spec, only=["dummy", "linear"])
    return spec


# ------------------------------------------------------------------ formulas

def test_formula_parser_allows_math_and_blocks_code():
    tree, refs = parse_formula('log1p(a) / (b + 1) + col("monthly spend") * where(a > 2, 1, 0)')
    assert refs == ["a", "b", "monthly spend"]
    frame = pd.DataFrame({"a": [0.0, 3.0], "b": [-1.0, 1.0], "monthly spend": [2.0, 4.0]})
    out = eval_formula(tree, frame)
    assert np.isnan(out[0])  # division by zero becomes NaN, not inf
    assert np.isclose(out[1], np.log1p(3) / 2 + 4)

    for bad in ["__import__('os').system('x')", "a.__class__", "(lambda: 1)()", "'text' + a", "[a, b]",
                "open('f')", "a if b else 1", "np.log(a)", "log(a, base=2)"]:
        try:
            parse_formula(bad)
            raise AssertionError(f"accepted {bad!r}")
        except FormulaError:
            pass


# ------------------------------------------------------------------ validation

def test_validation_tracks_columns_step_by_step():
    kinds = {"a": "numeric", "b": "numeric", "shop": "categorical", "day": "datetime"}
    ok = validate_recipe([
        {"type": "formula", "params": {"name": "ratio", "expression": "a / b"}},
        {"type": "log1p", "params": {"columns": ["ratio"]}},
        {"type": "group_agg", "params": {"by": "shop", "columns": ["ratio"], "stats": ["mean", "count"]}},
        {"type": "datetime_parts", "params": {"columns": ["day"]}},
        {"type": "drop", "params": {"columns": ["a"]}},
    ], kinds)
    assert ok["errors"] == [], ok["errors"]
    names = {c["name"] for c in ok["output"]}
    assert {"ratio", "ratio__by_shop__mean", "shop__count", "day__month"} <= names and "a" not in names
    assert {c["name"] for c in ok["columns_before"][1]} >= {"ratio"}

    bad = validate_recipe([
        {"type": "log1p", "params": {"columns": ["ratio"]}},             # used before it exists
        {"type": "target_encode", "params": {"columns": ["a"]}},          # numeric, needs categorical
        {"type": "drop", "params": {"columns": ["b"]}},
        {"type": "formula", "params": {"name": "x", "expression": "b * 2"}},  # dropped above
        {"type": "nope", "params": {}},
    ], kinds)
    assert [e["step"] for e in bad["errors"]] == [0, 1, 3, 4]

    multi = validate_recipe([{"type": "target_encode", "params": {"columns": ["shop"]}}], kinds, "multiclass", 3)
    assert {"shop__te_0", "shop__te_2"} <= {c["name"] for c in multi["output"]}


# ------------------------------------------------------------------ transformer

def test_target_encoding_is_cross_fitted_and_transform_handles_unseen():
    rng = np.random.default_rng(1)
    n = 600
    X = pd.DataFrame({"id_like": [f"k{i}" for i in rng.integers(0, 500, n)]}).astype(object)
    y = rng.integers(0, 2, n)  # pure noise
    recipe = FeatureRecipe([{"type": "target_encode", "params": {"columns": ["id_like"], "smoothing": 1}}], "binary")
    encoded = recipe.fit_transform(X, y)["id_like__te"].to_numpy()
    # In-sample encoding with smoothing 1 would correlate strongly with y; cross-fitted must not.
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, encoded) < 0.6
    unseen = recipe.transform(pd.DataFrame({"id_like": ["never-seen"]}).astype(object))
    assert np.isclose(unseen["id_like__te"].iloc[0], y.mean())


def test_every_step_runs_and_transform_matches_fit_columns():
    rng = np.random.default_rng(2)
    n = 200
    X = pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.exponential(size=n), "day": rng.uniform(19000, 19500, n),
        "shop": rng.choice(list("xyz"), n).astype(object), "plan": rng.choice(["basic", "pro"], n).astype(object),
    })
    X.loc[::7, "a"] = np.nan
    steps = [
        {"type": "formula", "params": {"name": "ab", "expression": "a * b"}},
        {"type": "log1p", "params": {"columns": ["b"]}},
        {"type": "clip", "params": {"columns": ["ab"], "lower": 0.05, "upper": 0.95}},
        {"type": "missing_flags", "params": {"columns": ["a"]}},
        {"type": "row_stats", "params": {"columns": ["a", "b"], "prefix": "ab_row"}},
        {"type": "datetime_parts", "params": {"columns": ["day"]}},
        {"type": "frequency_encode", "params": {"columns": ["shop"]}},
        {"type": "target_encode", "params": {"columns": ["plan"]}},
        {"type": "group_agg", "params": {"by": "shop", "columns": ["b"], "stats": ["mean", "std", "count", "diff_from_mean"]}},
        {"type": "combine", "params": {"first": "shop", "second": "plan"}},
        {"type": "drop", "params": {"columns": ["plan"]}},
    ]
    kinds = {"a": "numeric", "b": "numeric", "day": "datetime", "shop": "categorical", "plan": "categorical"}
    check = validate_recipe(steps, kinds)
    assert check["errors"] == [], check["errors"]
    recipe = FeatureRecipe(steps, "regression")
    fitted = recipe.fit_transform(X.iloc[:150], rng.normal(size=150))
    assert list(fitted.columns) == [c["name"] for c in check["output"]]  # validation predicts the real output
    later = X.iloc[150:].copy()
    later.loc[later.index[0], "shop"] = "unseen"
    out = recipe.transform(later)
    assert list(out.columns) == list(fitted.columns)
    assert out["shop__freq"].iloc[0] == 0.0 and np.isnan(out["b__by_shop__mean"].iloc[0])
    assert out["day__month"].between(1, 12).all()


# ------------------------------------------------------------------ experiments

def test_experiment_beats_control_and_leads_leaderboard():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        spec = _setup(root)
        good = create_experiment(root, "Closeness of a and b", "linear",
                                 [{"type": "formula", "params": {"name": "log_ratio", "expression": "abs(log(a / b))"}}],
                                 hypothesis="The target depends on how close a is to b.")
        junk = create_experiment(root, "Noise encoding", "linear",
                                 [{"type": "target_encode", "params": {"columns": ["noise_id"]}}])
        good_result = run_experiment(root, spec, good["id"])["result"]
        junk_result = run_experiment(root, spec, junk["id"])["result"]

        assert good_result["vs_control"]["verdict"] == "better", good_result["vs_control"]
        assert good_result["new_columns"][0]["column"] == "log_ratio"
        assert good_result["new_columns"][0]["strength"] > 0.9
        assert junk_result["vs_control"]["verdict"] in ("no_clear_change", "worse")
        assert (root / "experiments" / good["id"] / "submission.csv").exists()

        board = build_leaderboard(root)
        assert board["ready"] and not board["baselines_stale"]
        assert board["entries"][0]["id"] == f"experiment:{good['id']}"
        assert [e["rank"] for e in board["entries"]] == list(range(1, len(board["entries"]) + 1))
        assert "fold_scores" not in board["entries"][0]
        assert len(list_experiments(root)) == 2


def test_changed_settings_make_results_stale():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        spec = _setup(root)
        exp = create_experiment(root, "Ratio", "linear",
                                [{"type": "formula", "params": {"name": "r", "expression": "a / b"}}])
        run_experiment(root, spec, exp["id"])

        changed = ChallengeSpec(target="target", id_col="row_id", drop_cols=["junk"])
        run_analysis(root, changed)
        board = build_leaderboard(root)
        assert board["baselines_stale"] and board["entries"] == []
        assert {e["id"] for e in board["stale"]} >= {f"experiment:{exp['id']}", "baseline:linear"}
        try:
            run_experiment(root, changed, exp["id"])
            raise AssertionError("must refuse to compare against stale baselines")
        except RuntimeError as exc:
            assert "Re-run" in str(exc)

        run_baselines(root, changed, only=["dummy", "linear"])
        run_experiment(root, changed, exp["id"])
        board = build_leaderboard(root)
        assert not board["baselines_stale"] and board["stale"] == []
