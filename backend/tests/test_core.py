"""Core tests on synthetic data with planted problems the checks must find."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from quest_core import ChallengeSpec, run_analysis, run_baselines
from quest_core.cv_designer import adversarial_holdout_split, design_cv, make_splits
from quest_core.shift import psi
from quest_core.univariate import univariate_strength

FAST = ["dummy", "linear", "hist_gb"]


def _binary_data(seed=0, n=600, n_test=300):
    rng = np.random.default_rng(seed)

    def frame(m, shift=0.0):
        x1 = rng.normal(shift, 1, m)
        x2 = rng.normal(0, 1, m)
        city = rng.choice(["a", "b", "c", None], m)
        logit = 1.5 * x1 - x2 + (city == "a") * 0.8
        y = (rng.random(m) < 1 / (1 + np.exp(-logit))).astype(int)
        return pd.DataFrame({"row_id": np.arange(m), "x1": x1, "x2": x2, "city": city, "target": y})

    train = frame(n)
    test = frame(n_test, shift=1.5).drop(columns="target")  # planted covariate shift on x1
    test["row_id"] += n
    train["leak"] = train["target"] + rng.normal(0, 0.01, n)  # planted target proxy
    test["leak"] = rng.normal(0.5, 0.5, n_test)
    return train, test


def _write(tmp: Path, train: pd.DataFrame, test: pd.DataFrame) -> Path:
    train.to_csv(tmp / "train.csv", index=False)
    test.to_csv(tmp / "test.csv", index=False)
    return tmp


def test_binary_pipeline_finds_leak_and_shift():
    with tempfile.TemporaryDirectory() as d:
        root = _write(Path(d), *_binary_data())
        spec = ChallengeSpec(target="target", id_col="row_id")
        report = run_analysis(root, spec)

        assert report["task"] == "binary"
        assert report["metric"]["name"] == "roc_auc"
        codes = {f["code"]: f for f in report["leakage"]["findings"]}
        assert "target_proxy" in codes and "leak" in codes["target_proxy"]["columns"]
        assert report["shift"]["adversarial"]["band"] in ("moderate", "severe")
        assert "x1" in report["shift"]["drifting_features"]
        plan = report["cv_plan"]
        assert plan["strategy"] == "repeated_stratified_kfold"
        assert plan["n_repeats"] == 5  # tiny dataset gets repeats
        assert plan["adversarial_holdout"] is True
        assert (root / "likeness.npy").exists()
        json.loads((root / "analysis.json").read_text())  # valid JSON, no NaN tokens

        result = run_baselines(root, spec, only=FAST)
        models = {m["key"]: m for m in result["models"]}
        assert all(m["status"] == "ok" for m in models.values()), models
        assert result["n_folds"] == 25
        leader = result["models"][0]
        assert leader["rank"] == 1 and leader["vs_leader"]["mean_diff"] == 0.0
        assert models["dummy"]["cv_mean"] < 0.55  # prior model has no skill
        assert models["dummy"]["vs_leader"]["within_noise"] is False
        assert models["hist_gb"]["holdout_score"] is not None
        sub = pd.read_csv(root / "submissions" / "hist_gb.csv")
        assert list(sub.columns) == ["row_id", "target"] and len(sub) == 300
        assert sub["target"].between(0, 1).all()
        assert result["positive_class"] == 1
        assert (root / "runs.jsonl").read_text().count("\n") == 1 + len(FAST)


def test_regression_rmsle_with_dates():
    rng = np.random.default_rng(1)
    n = 800
    dates = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    x = rng.normal(0, 1, n)
    y = np.exp(1 + 0.5 * x + rng.normal(0, 0.2, n))
    train = pd.DataFrame({"date": dates, "x": x, "shop": rng.choice(list("pqrs"), n), "price": y})
    test = train.drop(columns="price").sample(200, random_state=0)
    with tempfile.TemporaryDirectory() as d:
        root = _write(Path(d), train, test)
        spec = ChallengeSpec(target="price", metric="rmsle")
        report = run_analysis(root, spec)
        assert report["task"] == "regression"
        kinds = {c["name"]: c["kind"] for c in report["profile"]["columns"]}
        assert kinds["date"] == "datetime" and kinds["shop"] == "categorical"
        assert report["profile"]["target"]["suggest_log_target"] in (True, False)
        result = run_baselines(root, spec, only=["dummy", "linear", "random_forest"])
        models = {m["key"]: m for m in result["models"]}
        assert all(m["status"] == "ok" for m in models.values()), models
        assert result["greater_is_better"] is False
        assert models["linear"]["cv_mean"] < models["dummy"]["cv_mean"]  # lower RMSLE is better
        assert result["models"][0]["key"] != "dummy"


def test_group_splits_keep_entities_together():
    rng = np.random.default_rng(2)
    n = 900
    groups = rng.integers(0, 60, n).astype(str)
    y = rng.integers(0, 3, n)
    plan = design_cv("multiclass", n, int(np.bincount(y).min()), has_groups=True, has_time=False, n_groups=60)
    assert plan["strategy"] == "stratified_group_kfold"
    for split in make_splits(plan, y, groups):
        assert not set(groups[split["train"]]) & set(groups[split["valid"]])

    reg_plan = design_cv("regression", n, None, has_groups=True, has_time=False, n_groups=60)
    assert reg_plan["strategy"] == "group_kfold"
    covered = np.zeros(n, dtype=int)
    for split in make_splits(reg_plan, rng.normal(size=n), groups):
        assert not set(groups[split["train"]]) & set(groups[split["valid"]])
        if split["repeat"] == 0:
            covered[split["valid"]] += 1
    assert (covered == 1).all()  # each row validated exactly once per repeat


def test_time_splits_move_forward():
    n = 500
    time = np.random.default_rng(3).permutation(n).astype(float)
    plan = design_cv("regression", n, None, has_groups=False, has_time=True)
    assert plan["strategy"] == "time_series" and plan["n_repeats"] == 1
    for split in make_splits(plan, np.zeros(n), time=time):
        assert time[split["train"]].max() < time[split["valid"]].min()


def test_rare_class_reduces_folds():
    plan = design_cv("multiclass", 400, 3, has_groups=False, has_time=False)
    assert plan["n_splits"] == 3 and plan["warnings"]
    single = design_cv("multiclass", 400, 1, has_groups=False, has_time=False)
    assert single["strategy"] == "repeated_kfold"


def test_psi_and_univariate_signals():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000), numeric=True) < 0.02
    assert psi(a, rng.normal(1.5, 1, 5000), numeric=True) > 0.5
    cats = rng.choice(list("abc"), 5000).astype(object)
    assert psi(cats, rng.choice(list("aaab"), 5000).astype(object), numeric=False) > 0.25

    # High-cardinality categorical that determines the label: needs target encoding to be caught.
    codes = rng.integers(0, 400, 4000).astype(float)
    label = (codes % 2).astype(int)
    assert univariate_strength(codes, label, "binary", categorical=True) > 0.95
    assert univariate_strength(rng.normal(size=4000), label, "binary") < 0.6


def test_holdout_split_takes_most_test_like_rows():
    likeness = np.array([0.1, 0.9, 0.2, 0.8, np.nan, 0.3, 0.95, 0.05, 0.4, 0.5])
    tr, va = adversarial_holdout_split(likeness, 0.2)
    assert set(va) == {1, 6} and len(tr) == 8 and 4 in tr


def test_id_like_ignores_continuous_floats():
    from quest_core.leakage import _is_id_like

    rng = np.random.default_rng(5)
    assert not _is_id_like(pd.Series(rng.normal(size=500)))
    assert _is_id_like(pd.Series(np.arange(500)))
    assert _is_id_like(pd.Series([f"cust-{i}" for i in range(500)]))
    assert not _is_id_like(pd.Series(rng.integers(0, 10, 500)))
