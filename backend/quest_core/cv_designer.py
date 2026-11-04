"""Choose a validation scheme that mirrors how the private test set was drawn.

Every decision carries a plain-language reason so the plan can be reviewed, not just obeyed.
Splits are generated once and reused for every model, which makes paired comparisons valid.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit

TEST_LIKE_HOLDOUT_SHARE = 0.2


def _repeats_for(n: int) -> int:
    if n < 1_000:
        return 5
    if n < 5_000:
        return 3
    if n < 20_000:
        return 2
    return 1


def design_cv(
    task: str,
    n_train: int,
    min_class_count: int | None,
    has_groups: bool,
    has_time: bool,
    shift: dict | None = None,
    leakage: dict | None = None,
    n_groups: int | None = None,
) -> dict:
    reasons: list[str] = []
    warnings: list[str] = []
    classification = task != "regression"
    n_splits = 5
    test_after_train = bool(shift and shift.get("time", {}).get("test_after_train"))

    if has_time:
        strategy = "time_series"
        n_repeats = 1
        reasons.append(
            "A time column is set, so folds move forward in time: each model trains on the past and "
            "validates on the period right after it, the way the private test set will be scored."
        )
        if shift and "time" in shift and not test_after_train:
            warnings.append(
                f"Test times overlap the train period ({shift['time']['overlap_share']:.0%} of test rows fall "
                "inside it). If the split was random rather than by date, a grouped or shuffled scheme may fit better."
            )
    elif has_groups:
        strategy = "stratified_group_kfold" if classification else "group_kfold"
        n_repeats = _repeats_for(n_train)
        if n_groups is not None and n_groups < n_splits:
            n_splits = max(2, n_groups)
            warnings.append(f"Only {n_groups} groups exist, so folds are reduced to {n_splits}.")
        reasons.append(
            "A group column is set, so every row of an entity stays on one side of each split. "
            "Otherwise the model memorises entities and CV overstates how it will do on unseen ones."
        )
    else:
        strategy = "repeated_stratified_kfold" if classification else "repeated_kfold"
        n_repeats = _repeats_for(n_train)
        reasons.append(
            "No time or group structure is declared, so folds are shuffled"
            + (" and stratified to keep class shares equal in every fold." if classification else ".")
        )

    if strategy != "time_series" and n_repeats > 1:
        reasons.append(
            f"With {n_train:,} rows a single {n_splits}-fold run is noisy, so it is repeated {n_repeats}× "
            "with different shuffles. Scores are reported as mean ± std over all folds."
        )

    if classification and min_class_count is not None and strategy in ("repeated_stratified_kfold", "stratified_group_kfold"):
        if min_class_count < 2:
            strategy = "repeated_kfold" if strategy == "repeated_stratified_kfold" else "group_kfold"
            warnings.append("A class has a single example, so stratification is impossible and was turned off.")
        elif min_class_count < n_splits:
            n_splits = max(2, min_class_count)
            warnings.append(f"The rarest class has {min_class_count} rows, so folds are reduced to {n_splits}.")

    adversarial_holdout = False
    if shift and shift.get("adversarial", {}).get("band") in ("moderate", "severe"):
        adversarial_holdout = True
        auc = shift["adversarial"]["auc"]
        warnings.append(
            f"Train and test are distinguishable (adversarial AUC {auc:.2f}). Random folds may overstate private-LB "
            f"scores, so each model is also scored on a test-like holdout: the {TEST_LIKE_HOLDOUT_SHARE:.0%} of train "
            "rows that look most like test. Weight that score heavily when choosing final submissions."
        )

    if leakage:
        codes = {f["code"] for f in leakage.get("findings", [])}
        if "row_order_signal" in codes and strategy != "time_series":
            warnings.append("Row order predicts the target. If the file is sorted by time, set the time column and re-run.")
        if "possible_groups" in codes and strategy not in ("group_kfold", "stratified_group_kfold"):
            warnings.append("A possible group column was detected. If entities repeat, set it and re-run.")
        if "train_duplicates" in codes and strategy not in ("group_kfold", "stratified_group_kfold"):
            warnings.append("Duplicate rows can land on both sides of a fold and inflate CV slightly.")

    return {
        "strategy": strategy,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "n_folds_total": n_splits * n_repeats,
        "adversarial_holdout": adversarial_holdout,
        "holdout_share": TEST_LIKE_HOLDOUT_SHARE if adversarial_holdout else None,
        "reasons": reasons,
        "warnings": warnings,
    }


def _group_kfold(groups: np.ndarray, n_splits: int, rng: np.random.Generator):
    """Shuffled group K-fold that balances fold sizes greedily (works on any sklearn version)."""
    unique, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-counts[order], kind="stable")]
    fold_of_group = np.empty(len(unique), dtype=int)
    sizes = np.zeros(n_splits, dtype=int)
    for g in order:
        f = int(np.argmin(sizes))
        fold_of_group[g] = f
        sizes[f] += counts[g]
    row_fold = fold_of_group[np.searchsorted(unique, groups)]
    for f in range(n_splits):
        yield np.where(row_fold != f)[0], np.where(row_fold == f)[0]


def make_splits(
    plan: dict,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    time: np.ndarray | None = None,
    seed: int = 42,
) -> list[dict]:
    """Materialise the plan as a list of {repeat, fold, train, valid} index arrays."""
    n = len(y)
    X_dummy = np.zeros((n, 1))
    splits = []
    strategy, k = plan["strategy"], plan["n_splits"]

    if strategy == "time_series":
        order = np.argsort(np.asarray(time, dtype=float), kind="stable")
        for f, (tr, va) in enumerate(TimeSeriesSplit(n_splits=k).split(order)):
            splits.append({"repeat": 0, "fold": f, "train": order[tr], "valid": order[va]})
        return splits

    for r in range(plan["n_repeats"]):
        rs = seed + r
        if strategy == "repeated_stratified_kfold":
            it = StratifiedKFold(k, shuffle=True, random_state=rs).split(X_dummy, y)
        elif strategy == "repeated_kfold":
            it = KFold(k, shuffle=True, random_state=rs).split(X_dummy)
        elif strategy == "stratified_group_kfold":
            it = StratifiedGroupKFold(k, shuffle=True, random_state=rs).split(X_dummy, y, groups)
        elif strategy == "group_kfold":
            it = _group_kfold(np.asarray(groups), k, np.random.default_rng(rs))
        else:
            raise ValueError(f"Unknown CV strategy '{strategy}'.")
        for f, (tr, va) in enumerate(it):
            splits.append({"repeat": r, "fold": f, "train": np.asarray(tr), "valid": np.asarray(va)})
    return splits


def adversarial_holdout_split(likeness: np.ndarray, share: float = TEST_LIKE_HOLDOUT_SHARE) -> tuple[np.ndarray, np.ndarray]:
    """Train on the least test-like rows, validate on the most test-like ones."""
    scores = np.nan_to_num(likeness, nan=-1.0)
    n_hold = max(1, int(round(len(scores) * share)))
    order = np.argsort(scores, kind="stable")
    return np.sort(order[:-n_hold]), np.sort(order[-n_hold:])
