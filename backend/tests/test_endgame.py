"""Week-4 tests: consistent candidate scoring, public-LB noise, honest blends, tracking and final picks."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from quest_core import run_baselines, run_experiment
from quest_core import endgame as eg
from quest_core.experiments import create_experiment
from tests.test_experiments import _setup


def _with_experiment(root: Path):
    spec = _setup(root)
    run_baselines(root, spec, only=["dummy", "linear", "hist_gb"])
    exp = create_experiment(root, "Closeness", "linear",
                            [{"type": "formula", "params": {"name": "c", "expression": "abs(log(a / b))"}}])
    run_experiment(root, spec, exp["id"])
    return spec, exp


def test_greedy_weights_overfit_in_sample_but_not_when_scored_honestly():
    """Blending pure-noise predictors: in-sample selection looks like a gain; honest per-fold scoring doesn't."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    preds = [np.column_stack([1 - p, p]) for p in rng.uniform(size=(12, n))]
    from quest_core.spec import METRICS

    metric = METRICS["roc_auc"]
    all_rows = np.arange(n)
    w = eg.greedy_weights(preds, y, metric, all_rows, 1)
    in_sample = roc_auc_score(y, eg.combine(preds, w, "mean")[:, 1])
    honest = []
    for tr, va in KFold(5, shuffle=True, random_state=0).split(all_rows):
        wf = eg.greedy_weights(preds, y, metric, tr, 1)
        honest.append(roc_auc_score(y[va], eg.combine([p[va] for p in preds], wf, "mean")[:, 1]))
    assert np.isclose(w.sum(), 1) and in_sample > 0.56
    assert abs(np.mean(honest) - 0.5) < 0.05


def test_endgame_summary_noise_blend_tracking_and_picks():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        spec, exp = _with_experiment(root)

        summary = eg.endgame_summary(root, spec)
        assert summary["ready"] and summary["noise"] is None
        ids = [c["id"] for c in summary["candidates"]]
        assert ids[0] == f"experiment:{exp['id']}" and "baseline:dummy" not in ids
        assert len(summary["picks"]["picks"]) == 2 and summary["picks"]["picks"][0]["reasons"]

        eg.save_settings(root, 0.3, 2)
        noise = eg.endgame_summary(root, spec)["noise"]
        assert noise["n_public"] == 60 and noise["n_private"] == 140  # 30% of 200 test rows
        assert noise["public_sd"] > noise["private_sd"] > 0  # fewer rows, more luck
        assert noise["paired_public_sd"] > 0

        blend = eg.create_blend(root, "Closeness + GBM", [f"experiment:{exp['id']}", "baseline:hist_gb"], "greedy")
        result = eg.run_blend(root, spec, blend["id"])["result"]
        assert np.isclose(sum(w["weight"] for w in result["weights"]), 1)
        assert result["vs_best_member"]["verdict"] in ("better", "no_clear_change", "worse")
        assert result["best_member"]["id"] == f"experiment:{exp['id']}"
        assert eg.blend_submission_path(root, blend["id"]).exists()
        rank = eg.create_blend(root, "Rank", [f"experiment:{exp['id']}", "baseline:linear"], "rank_mean")
        assert eg.run_blend(root, spec, rank["id"])["result"]["status"] == "ok"

        summary = eg.endgame_summary(root, spec)
        assert any(c["kind"] == "blend" for c in summary["candidates"])
        assert len(summary["correlations"]["matrix"]) == len(summary["correlations"]["labels"])

        by_id = {c["id"]: c for c in summary["candidates"]}
        for entry, public in [(f"experiment:{exp['id']}", None), ("baseline:hist_gb", None), ("baseline:linear", 0.99)]:
            cv = by_id[entry]["cv_mean"]
            eg.add_submission(root, entry, by_id[entry]["label"], public if public is not None else cv - 0.01, "", cv)
        summary = eg.endgame_summary(root, spec)
        rows = {r["entry_id"]: r for r in summary["tracking"]["rows"]}
        assert rows["baseline:linear"]["flag"] == "lucky"  # public far above what its CV predicts
        assert summary["tracking"]["best_public"] == rows["baseline:linear"]["id"]
        assert any("best public score" in n for n in summary["picks"]["notes"])
        eg.delete_submission(root, rows["baseline:linear"]["id"])
        assert len(eg.list_submissions(root)) == 2


def test_rank_blend_refused_for_calibrated_metrics():
    from quest_core import ChallengeSpec, run_analysis
    from tests.test_experiments import _ratio_data

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        train, test = _ratio_data()
        train.to_csv(root / "train.csv", index=False)
        test.to_csv(root / "test.csv", index=False)
        spec = ChallengeSpec(target="target", id_col="row_id", metric="log_loss")
        run_analysis(root, spec)
        run_baselines(root, spec, only=["dummy", "linear", "hist_gb"])
        blend = eg.create_blend(root, "Rank", ["baseline:linear", "baseline:hist_gb"], "rank_mean")
        try:
            eg.run_blend(root, spec, blend["id"])
            raise AssertionError("rank blend must be refused for log loss")
        except ValueError as exc:
            assert "ranking metrics" in str(exc)


def test_picks_handle_a_blend_leader_under_shift():
    """A blend has no holdout score; under shift the picks must not crash or switch on a missing comparison."""
    ctx = {"sign": 1, "plan": {"n_splits": 5}, "analysis": {"shift": {"adversarial": {"band": "severe"}}}}
    rng = np.random.default_rng(3)
    base = rng.uniform(size=200)

    def cand(cid, kind, mean, holdout, noise):
        return {"id": cid, "kind": kind, "label": cid, "cv_mean": mean, "cv_se": 0.002, "holdout": holdout,
                "fold_scores": mean + rng.normal(0, 0.002, 25), "submission": "/x",
                "oof_array": np.column_stack([1 - base, np.clip(base + rng.normal(0, noise, 200), 0, 1)])}

    cands = [cand("blend:1", "blend", 0.95, None, 0.01), cand("experiment:a", "experiment", 0.949, 0.90, 0.05),
             cand("baseline:b", "baseline", 0.90, 0.85, 0.2)]
    out = eg.recommend_picks(ctx, cands, 2, None, {"rows": [], "best_public": None, "spearman": None})
    assert out["picks"][0]["id"] == "blend:1"
    assert any("no test-like holdout" in r for r in out["picks"][0]["reasons"])
    assert out["picks"][1]["id"] == "experiment:a"
