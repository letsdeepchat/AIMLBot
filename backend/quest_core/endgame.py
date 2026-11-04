"""Endgame: public-leaderboard noise, submission tracking, honest blends and final picks.

Every candidate (baseline, experiment, blend) is scored here the same way: from its saved out-of-fold
predictions on each validation fold of the shared plan. Mixing fold-model scores with OOF-based blend
scores would favour blends, because averaged OOF predictions are smoother than any single fold model.

Blend weights are fitted on the other folds' rows and scored on the held-out fold, so a blend's score is
not inflated by choosing and judging its weights on the same rows.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from . import baselines as bl
from .cv_designer import make_splits
from .experiments import list_experiments
from .pipeline import _write_json, load_inputs
from .spec import ChallengeSpec

Stage = Callable[[str], None]
BLEND_METHODS = {
    "mean": "Equal-weight average",
    "rank_mean": "Rank average (ranking metrics only)",
    "greedy": "Greedy ensemble selection (weights fitted per fold)",
}
RANK_METRICS = {"roc_auc"}
GREEDY_ITERATIONS = 25


def _root(challenge_dir) -> Path:
    return Path(challenge_dir) / "endgame"


# ------------------------------------------------------------------ settings and submissions


def load_settings(challenge_dir) -> dict:
    path = _root(challenge_dir) / "settings.json"
    defaults = {"public_share": None, "n_picks": 2}
    return {**defaults, **json.loads(path.read_text())} if path.exists() else defaults


def save_settings(challenge_dir, public_share: float | None, n_picks: int) -> dict:
    if public_share is not None and not 0 < public_share < 1:
        raise ValueError("The public share must be between 0 and 1, e.g. 0.3 for 30%.")
    if not 1 <= n_picks <= 5:
        raise ValueError("Choose between 1 and 5 final picks.")
    _root(challenge_dir).mkdir(parents=True, exist_ok=True)
    settings = {"public_share": public_share, "n_picks": int(n_picks)}
    _write_json(_root(challenge_dir) / "settings.json", settings)
    return settings


def list_submissions(challenge_dir) -> list[dict]:
    path = _root(challenge_dir) / "submissions.json"
    return json.loads(path.read_text()) if path.exists() else []


def add_submission(challenge_dir, entry_id: str, label: str, public_score: float, note: str, cv_snapshot) -> dict:
    subs = list_submissions(challenge_dir)
    record = {"id": uuid.uuid4().hex[:8], "entry_id": entry_id, "label": label, "public_score": float(public_score),
              "note": note.strip(), "cv_snapshot": cv_snapshot, "submitted_at": time.time()}
    _root(challenge_dir).mkdir(parents=True, exist_ok=True)
    (_root(challenge_dir) / "submissions.json").write_text(json.dumps(subs + [record], indent=2))
    return record


def delete_submission(challenge_dir, submission_id: str) -> None:
    subs = [s for s in list_submissions(challenge_dir) if s["id"] != submission_id]
    (_root(challenge_dir) / "submissions.json").write_text(json.dumps(subs, indent=2))


# ------------------------------------------------------------------ candidates


def _context(challenge_dir: Path, spec: ChallengeSpec) -> dict:
    root = Path(challenge_dir)
    analysis = json.loads((root / "analysis.json").read_text())
    baselines = json.loads((root / "baselines.json").read_text())
    inp = load_inputs(root, spec)
    plan = analysis["cv_plan"]
    splits = make_splits(plan, inp.y, inp.groups,
                         inp.time_train.to_numpy() if inp.time_train is not None else None, spec.seed)
    return {"root": root, "analysis": analysis, "baselines": baselines, "inp": inp, "plan": plan, "splits": splits,
            "fingerprint": analysis.get("fingerprint"), "metric": inp.metric,
            "sign": 1 if inp.metric.greater_is_better else -1}


def candidate_entries(challenge_dir: Path, fingerprint: str | None, baselines: dict) -> list[dict]:
    """Every current entry whose out-of-fold and test predictions are on disk."""
    root = Path(challenge_dir)
    out = []
    if baselines.get("fingerprint") == fingerprint:
        for m in baselines["models"]:
            if m["status"] == "ok" and m["key"] != "dummy" and (root / "preds" / f"{m['key']}_oof.npy").exists():
                out.append({"id": f"baseline:{m['key']}", "kind": "baseline", "label": m["label"],
                            "holdout": m.get("holdout_score"),
                            "oof": root / "preds" / f"{m['key']}_oof.npy", "test": root / "preds" / f"{m['key']}_test.npy",
                            "submission": f"/submissions/{m['key']}"})
    for e in list_experiments(root):
        r = e.get("result") or {}
        folder = root / "experiments" / e["id"]
        if r.get("status") == "ok" and e.get("fingerprint") == fingerprint and (folder / "oof.npy").exists():
            out.append({"id": f"experiment:{e['id']}", "kind": "experiment", "label": e["name"],
                        "sublabel": r.get("model_label"), "holdout": r.get("holdout_score"),
                        "oof": folder / "oof.npy", "test": folder / "test.npy",
                        "submission": f"/experiments/{e['id']}/submission"})
    for b in list_blends(root):
        r = b.get("result") or {}
        folder = root / "endgame" / "blends" / b["id"]
        if r.get("status") == "ok" and b.get("fingerprint") == fingerprint and (folder / "oof.npy").exists():
            out.append({"id": f"blend:{b['id']}", "kind": "blend", "label": b["name"],
                        "sublabel": BLEND_METHODS[b["method"]].split(" (")[0], "holdout": None,
                        "oof": folder / "oof.npy", "test": folder / "test.npy",
                        "submission": f"/endgame/blends/{b['id']}/submission"})
    return out


def _covered(oof: np.ndarray) -> np.ndarray:
    return ~np.isnan(oof).any(axis=1) if oof.ndim == 2 else ~np.isnan(oof)


def oof_fold_scores(oof: np.ndarray, y: np.ndarray, metric, splits: list[dict]) -> np.ndarray:
    covered = _covered(oof)
    scores = []
    for split in splits:
        va = split["valid"][covered[split["valid"]]]
        scores.append(bl._safe_score(metric, y[va], oof[va]) if len(va) else np.nan)
    return np.asarray(scores, dtype=float)


def _mean_se(scores: np.ndarray, n_splits: int) -> tuple[float | None, float | None]:
    s = scores[np.isfinite(scores)]
    if not len(s):
        return None, None
    if len(s) < 2:
        return float(s.mean()), None
    return float(s.mean()), float(np.sqrt((1 / len(s) + 1 / max(n_splits - 1, 1)) * s.var(ddof=1)))


def _signal(pred: np.ndarray) -> np.ndarray:
    """One column per prediction for correlation: the positive-class probability, or the value."""
    if pred.ndim == 2:
        return pred[:, 1] if pred.shape[1] == 2 else pred.reshape(-1)
    return pred


def score_candidates(ctx: dict, entries: list[dict]) -> list[dict]:
    y, metric, splits, n_splits = ctx["inp"].y, ctx["metric"], ctx["splits"], ctx["plan"]["n_splits"]
    out = []
    for e in entries:
        oof = np.load(e["oof"])
        scores = oof_fold_scores(oof, y, metric, splits)
        mean, se = _mean_se(scores, n_splits)
        if mean is None:
            continue
        out.append({**e, "oof_array": oof, "fold_scores": scores, "cv_mean": mean, "cv_se": se})
    out.sort(key=lambda c: ctx["sign"] * c["cv_mean"], reverse=True)
    return out


def correlations(cands: list[dict], limit: int = 8) -> dict:
    top = cands[:limit]
    if len(top) < 2:
        return {"labels": [c["label"] for c in top], "ids": [c["id"] for c in top], "matrix": [[1.0]] if top else []}
    mask = np.all([_covered(c["oof_array"]) for c in top], axis=0)
    signals = np.vstack([_signal(c["oof_array"])[mask] for c in top])
    matrix = np.corrcoef(signals)
    return {"labels": [c["label"] for c in top], "ids": [c["id"] for c in top],
            "matrix": [[round(float(v), 4) for v in row] for row in matrix]}


# ------------------------------------------------------------------ public leaderboard noise


def public_noise(ctx: dict, cands: list[dict], public_share: float, n_boot: int = 300, seed: int = 0) -> dict | None:
    """How much a score moves on the public and private splits by luck alone, from bootstrapped OOF rows.

    Also: the luck-only spread of the public gap between the two best candidates, i.e. the smallest public
    difference between similar models that means anything.
    """
    if not cands or not public_share:
        return None
    y, metric = ctx["inp"].y, ctx["metric"]
    n_test = len(ctx["inp"].X_test)
    n_public = max(2, int(round(n_test * public_share)))
    n_private = max(2, n_test - n_public)
    rng = np.random.default_rng(seed)
    best = cands[0]
    pool = np.where(_covered(best["oof_array"]))[0]
    other = cands[1] if len(cands) > 1 else None
    if other is not None:
        pool = np.intersect1d(pool, np.where(_covered(other["oof_array"]))[0])

    def spread(n, paired=False):
        values = []
        for _ in range(n_boot):
            idx = rng.choice(pool, n, replace=n > len(pool))
            a = bl._safe_score(metric, y[idx], best["oof_array"][idx])
            if paired:
                a -= bl._safe_score(metric, y[idx], other["oof_array"][idx])
            if np.isfinite(a):
                values.append(a)
        return float(np.std(values, ddof=1)) if len(values) > 10 else None

    public_sd, private_sd = spread(n_public), spread(n_private)
    return {
        "public_share": public_share, "n_public": n_public, "n_private": n_private,
        "public_sd": public_sd, "private_sd": private_sd,
        "public_private_gap_sd": float(np.hypot(public_sd, private_sd)) if public_sd and private_sd else None,
        "pair": [best["label"], other["label"]] if other is not None else None,
        "paired_public_sd": spread(n_public, paired=True) if other is not None else None,
        "rows_in_oof": int(len(pool)),
    }


# ------------------------------------------------------------------ submissions vs CV


def track_submissions(subs: list[dict], by_id: dict, noise: dict | None, sign: int) -> dict:
    rows = []
    for s in subs:
        current = by_id.get(s["entry_id"])
        cv = current["cv_mean"] if current else s.get("cv_snapshot")
        rows.append({**s, "cv": cv, "current": current is not None,
                     "gap": None if cv is None else s["public_score"] - cv})
    gaps = [r["gap"] for r in rows if r["gap"] is not None]
    typical = float(np.median(gaps)) if gaps else None
    sd = (noise or {}).get("public_sd")
    for r in rows:
        r["flag"] = None
        if typical is not None and sd and r["gap"] is not None and len(gaps) >= 3:
            surprise = (r["gap"] - typical) * sign  # positive = public kinder than CV predicts
            if abs(surprise) > 2 * sd:
                r["flag"] = "lucky" if surprise > 0 else "unlucky"
    paired = [(r["cv"], r["public_score"]) for r in rows if r["cv"] is not None]
    rho = None
    if len(paired) >= 3 and len({p[0] for p in paired}) > 1 and len({p[1] for p in paired}) > 1:
        rho = float(spearmanr([p[0] for p in paired], [p[1] for p in paired]).statistic)
    best = max(rows, key=lambda r: sign * r["public_score"]) if rows else None
    return {"rows": rows, "typical_gap": typical, "spearman": rho, "best_public": best["id"] if best else None}


# ------------------------------------------------------------------ blends


def list_blends(challenge_dir) -> list[dict]:
    folder = _root(challenge_dir) / "blends"
    if not folder.exists():
        return []
    return sorted((json.loads(p.read_text()) for p in folder.glob("*/blend.json")), key=lambda b: b["created_at"], reverse=True)


def create_blend(challenge_dir, name: str, members: list[str], method: str) -> dict:
    if method not in BLEND_METHODS:
        raise ValueError(f"Method must be one of: {', '.join(BLEND_METHODS)}.")
    if len(set(members)) < 2:
        raise ValueError("Pick at least two entries to blend.")
    blend = {"id": uuid.uuid4().hex[:10], "name": name.strip() or "Blend", "members": list(dict.fromkeys(members)),
             "method": method, "created_at": time.time(), "result": None}
    folder = _root(challenge_dir) / "blends" / blend["id"]
    folder.mkdir(parents=True)
    _write_json(folder / "blend.json", blend)
    return blend


def delete_blend(challenge_dir, blend_id: str) -> None:
    if not blend_id.isalnum():
        raise ValueError("Bad blend id.")
    shutil.rmtree(_root(challenge_dir) / "blends" / blend_id, ignore_errors=True)


def blend_submission_path(challenge_dir, blend_id: str) -> Path:
    if not blend_id.isalnum():
        raise ValueError("Bad blend id.")
    return _root(challenge_dir) / "blends" / blend_id / "submission.csv"


def _as_matrix(pred: np.ndarray) -> np.ndarray:
    return pred if pred.ndim == 2 else pred[:, None]


def combine(preds: list[np.ndarray], weights: np.ndarray, method: str) -> np.ndarray:
    if method == "rank_mean":
        ranks = [rankdata(_signal(p), nan_policy="omit") / np.sum(np.isfinite(_signal(p))) for p in preds]
        score = np.sum([w * r for w, r in zip(weights, ranks)], axis=0)
        return np.column_stack([1 - score, score])
    stacked = np.sum([w * _as_matrix(p) for w, p in zip(weights, preds)], axis=0)
    return stacked if preds[0].ndim == 2 else stacked[:, 0]


def greedy_weights(preds: list[np.ndarray], y: np.ndarray, metric, rows: np.ndarray, sign: int,
                   iterations: int = GREEDY_ITERATIONS) -> np.ndarray:
    """Caruana ensemble selection with replacement on `rows`, starting from the best single member."""
    counts = np.zeros(len(preds))
    singles = [sign * bl._safe_score(metric, y[rows], p[rows]) for p in preds]
    counts[int(np.nanargmax(singles))] = 1
    for _ in range(iterations - 1):
        best_gain, best_j = -np.inf, None
        for j in range(len(preds)):
            trial = counts.copy()
            trial[j] += 1
            score = sign * bl._safe_score(metric, y[rows], combine([p[rows] for p in preds], trial / trial.sum(), "mean"))
            if score > best_gain:
                best_gain, best_j = score, j
        counts[best_j] += 1
    return counts / counts.sum()


def run_blend(challenge_dir: Path, spec: ChallengeSpec, blend_id: str, on_stage: Stage | None = None) -> dict:
    stage = on_stage or (lambda _: None)
    root = Path(challenge_dir)
    folder = _root(root) / "blends" / blend_id
    blend = json.loads((folder / "blend.json").read_text())

    stage("Loading predictions")
    ctx = _context(root, spec)
    metric, sign, y, splits = ctx["metric"], ctx["sign"], ctx["inp"].y, ctx["splits"]
    if blend["method"] == "rank_mean" and metric.name not in RANK_METRICS:
        raise ValueError(f"Rank averaging only suits ranking metrics; {metric.label} needs calibrated predictions.")
    available = {e["id"]: e for e in candidate_entries(root, ctx["fingerprint"], ctx["baselines"])}
    missing = [m for m in blend["members"] if m not in available]
    if missing:
        raise RuntimeError(f"These entries are missing or stale: {', '.join(missing)}. Re-run them or pick others.")
    members = [available[m] for m in blend["members"]]
    oofs = [np.load(m["oof"]) for m in members]
    tests = [np.load(m["test"]) for m in members]
    covered = np.all([_covered(o) for o in oofs], axis=0)

    stage("Fitting weights on other folds and scoring each held-out fold")
    equal = np.full(len(members), 1 / len(members))
    blend_scores, member_scores = [], [[] for _ in members]
    honest_sum = np.zeros(_as_matrix(oofs[0]).shape if oofs[0].ndim == 2 else oofs[0].shape)
    honest_n = np.zeros(len(y))
    for split in splits:
        va = split["valid"][covered[split["valid"]]]
        fit_rows = np.setdiff1d(np.where(covered)[0], split["valid"])
        if not len(va) or not len(fit_rows):
            blend_scores.append(np.nan)
            for k in range(len(members)):
                member_scores[k].append(np.nan)
            continue
        w = greedy_weights(oofs, y, metric, fit_rows, sign) if blend["method"] == "greedy" else equal
        pred = combine([o[va] for o in oofs], w, blend["method"])
        blend_scores.append(bl._safe_score(metric, y[va], pred))
        for k, o in enumerate(oofs):
            member_scores[k].append(bl._safe_score(metric, y[va], o[va]))
        honest_sum[va] += pred
        honest_n[va] += 1

    blend_scores = np.asarray(blend_scores, dtype=float)
    member_means = [np.nanmean(s) for s in member_scores]
    best_k = int(np.nanargmax([sign * m for m in member_means]))
    comparison = bl.corrected_comparison(sign * (blend_scores - np.asarray(member_scores[best_k])), ctx["plan"]["n_splits"])
    verdict = "no_clear_change" if comparison["within_noise"] or comparison["mean_diff"] is None else (
        "better" if comparison["mean_diff"] > 0 else "worse")

    stage("Fitting final weights and blending test predictions")
    final_w = greedy_weights(oofs, y, metric, np.where(covered)[0], sign) if blend["method"] == "greedy" else equal
    honest = np.full(honest_sum.shape, np.nan)
    seen = honest_n > 0
    honest[seen] = honest_sum[seen] / (honest_n[seen] if honest_sum.ndim == 1 else honest_n[seen, None])
    test_pred = combine(tests, final_w, blend["method"])
    np.save(folder / "oof.npy", honest)
    np.save(folder / "test.npy", test_pred)
    inp = ctx["inp"]
    ids = inp.test[spec.id_col] if spec.id_col and spec.id_col in inp.test.columns else pd.Series(range(len(inp.test)))
    bl.build_submission(ids, test_pred, inp.task, metric, inp.class_labels, spec.target, spec.id_col or "id").to_csv(folder / "submission.csv", index=False)

    mean, se = _mean_se(blend_scores, ctx["plan"]["n_splits"])
    blend["result"] = {
        "status": "ok", "cv_mean": mean, "cv_se": se,
        "weights": [{"id": m["id"], "label": m["label"], "weight": round(float(w), 4)} for m, w in zip(members, final_w)],
        "best_member": {"id": members[best_k]["id"], "label": members[best_k]["label"], "cv_mean": float(member_means[best_k])},
        "vs_best_member": {**comparison, "verdict": verdict},
        "note": "Rank averages only preserve ordering; use them for AUC-style metrics." if blend["method"] == "rank_mean" else None,
        "finished_at": time.time(),
    }
    blend["fingerprint"] = ctx["fingerprint"]
    _write_json(folder / "blend.json", blend)
    return blend


# ------------------------------------------------------------------ final picks


def _paired(a: dict, b: dict, sign: int, n_splits: int) -> dict:
    return bl.corrected_comparison(sign * (a["fold_scores"] - b["fold_scores"]), n_splits)


def recommend_picks(ctx: dict, cands: list[dict], n_picks: int, noise: dict | None, tracking: dict) -> dict:
    if not cands:
        return {"picks": [], "notes": ["Run the baselines (and ideally a few experiments) to get candidates."]}
    sign, n_splits = ctx["sign"], ctx["plan"]["n_splits"]
    shifted = ctx["analysis"]["shift"]["adversarial"]["band"] in ("moderate", "severe")

    def robust(c):  # lower confidence bound in "higher is better" space
        return sign * c["cv_mean"] - (c["cv_se"] or 0)

    ranked = sorted(cands, key=robust, reverse=True)
    first = ranked[0]
    reasons = {first["id"]: ["Highest score once its fold-to-fold uncertainty is subtracted, so it's the safest single bet."]}
    if shifted and first["holdout"] is None:
        reasons[first["id"]].append(
            "Train and test differ here, and a blend has no test-like holdout score. Check that its members hold up "
            "on the holdout before relying on it.")
    elif shifted:
        tied = [c for c in ranked if c["holdout"] is not None and c["id"] != first["id"]
                and _paired(c, first, sign, n_splits)["within_noise"]]
        better = [c for c in tied if sign * c["holdout"] > sign * first["holdout"]]
        if better:
            by_holdout = max(better, key=lambda c: sign * c["holdout"])
            reasons[by_holdout["id"]] = [
                f"Statistically tied with {first['label']} on CV, and better on the test-like holdout "
                f"({by_holdout['holdout']:.4f} vs {first['holdout']:.4f}). Train and test differ here, so the holdout "
                "is the better guide to the private leaderboard."]
            first = by_holdout
    picks = [first]

    while len(picks) < min(n_picks, len(cands)):
        rest = [c for c in ranked if c["id"] not in {p["id"] for p in picks}]
        tied = [c for c in rest if _paired(c, picks[0], sign, n_splits)["within_noise"]]
        pool = tied or rest
        mask = np.all([_covered(c["oof_array"]) for c in picks + pool], axis=0)

        def max_corr(c):
            return max(float(np.corrcoef(_signal(c["oof_array"])[mask], _signal(p["oof_array"])[mask])[0, 1]) for p in picks)

        choice = min(pool, key=max_corr) if tied else pool[0]
        corr = max_corr(choice)
        reasons[choice["id"]] = (
            [f"Statistically tied with {picks[0]['label']}, and the least similar of the tied candidates "
             f"(prediction correlation {corr:.3f}), so it hedges against the private split favouring a different model."]
            if tied else
            [f"Nothing else is statistically tied with {picks[0]['label']}; this is the next best ({choice['cv_mean']:.4f}), "
             f"correlation {corr:.3f} with your other picks."])
        picks.append(choice)

    notes = []
    if noise and noise.get("paired_public_sd"):
        notes.append(f"On a public split of {noise['n_public']} rows, two similar models' public scores differ by up to "
                     f"±{2 * noise['paired_public_sd']:.4f} from luck alone. Smaller public gaps don't mean anything.")
    best_public = next((r for r in tracking["rows"] if r["id"] == tracking["best_public"]), None)
    if best_public and best_public["entry_id"] not in {p["id"] for p in picks}:
        notes.append(f"Your best public score is {best_public['label']} ({best_public['public_score']:.4f}). It isn't a pick: "
                     f"its CV ({best_public['cv']:.4f}) is behind your picks, so its public lead is most likely luck on the public rows.")
    if tracking.get("spearman") is not None and tracking["spearman"] < 0.3:
        notes.append(f"CV and public scores barely agree on ordering (Spearman {tracking['spearman']:.2f}). With a small "
                     "or shifted public split that's expected; it's a reason to trust CV, not to chase the public board.")
    return {"picks": [{"id": p["id"], "label": p["label"], "sublabel": p.get("sublabel"), "kind": p["kind"],
                       "cv_mean": p["cv_mean"], "cv_se": p["cv_se"], "holdout": p["holdout"],
                       "submission": p["submission"], "reasons": reasons.get(p["id"], [])} for p in picks],
            "notes": notes}


# ------------------------------------------------------------------ summary


def endgame_summary(challenge_dir: Path, spec: ChallengeSpec) -> dict:
    root = Path(challenge_dir)
    if not (root / "baselines.json").exists():
        return {"ready": False}
    ctx = _context(root, spec)
    if ctx["baselines"].get("fingerprint") != ctx["fingerprint"]:
        return {"ready": False, "stale": True}
    settings = load_settings(root)
    cands = score_candidates(ctx, candidate_entries(root, ctx["fingerprint"], ctx["baselines"]))
    noise = public_noise(ctx, cands, settings["public_share"]) if settings["public_share"] else None
    by_id = {c["id"]: c for c in cands}
    tracking = track_submissions(list_submissions(root), by_id, noise, ctx["sign"])
    picks = recommend_picks(ctx, cands, settings["n_picks"], noise, tracking)
    return {
        "ready": True, "settings": settings, "metric_label": ctx["metric"].label,
        "greater_is_better": ctx["metric"].greater_is_better, "n_test": len(ctx["inp"].X_test),
        "candidates": [{"id": c["id"], "kind": c["kind"], "label": c["label"], "sublabel": c.get("sublabel"),
                        "cv_mean": c["cv_mean"], "cv_se": c["cv_se"], "holdout": c["holdout"],
                        "submission": c["submission"]} for c in cands],
        "correlations": correlations(cands),
        "noise": noise, "tracking": tracking, "picks": picks,
        "blends": [{k: v for k, v in b.items()} for b in list_blends(root)],
        "blend_methods": BLEND_METHODS, "rank_ok": ctx["metric"].name in RANK_METRICS,
    }
