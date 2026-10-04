"""Evaluation metrics: macro-F1 headline, PR-AUC, fault-level bootstrap, McNemar.

Statistics are right-sized for small n_fault (CWRU ≈ 12). Friedman/Nemenyi
intentionally omitted — no power at n=12.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.stats import binom
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass
class EvalResult:
    macro_f1: float
    accuracy: float
    confusion: np.ndarray
    pr_auc: float | None = None
    roc_auc: float | None = None
    per_class_f1: dict[str, float] = field(default_factory=dict)
    majority_baseline_f1: float | None = None
    majority_baseline_acc: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "per_class_f1": self.per_class_f1,
            "majority_baseline_f1": self.majority_baseline_f1,
            "majority_baseline_acc": self.majority_baseline_acc,
            "confusion": self.confusion.tolist(),
            **self.extras,
        }


def majority_label(y_train: np.ndarray) -> int:
    """Most frequent TRAIN label; ties broken by lowest label (np.unique is sorted)."""
    vals, counts = np.unique(np.asarray(y_train), return_counts=True)
    if len(vals) == 0:
        raise ValueError("majority baseline needs at least one training label")
    return int(vals[np.argmax(counts)])


def majority_baseline_metrics(
    y_true: np.ndarray,
    y_train: np.ndarray | None = None,
    labels: Sequence[int] | None = None,
) -> tuple[float, float]:
    """(macro-F1, accuracy) of predicting the train-majority label on y_true.

    The baseline is FIT on ``y_train`` and SCORED on ``y_true``. Passing
    ``y_train=None`` scores the in-sample (resubstitution) baseline — only
    meaningful as a sanity check on a single vector, never for a CV fold,
    because it lets the baseline read the test labels.
    """
    y_true = np.asarray(y_true)
    if y_train is None:
        warnings.warn(
            "majority_baseline_metrics without y_train fits the baseline on the "
            "scored labels (in-sample); pass y_train for any held-out evaluation",
            stacklevel=2,
        )
        y_train = y_true
    pred = np.full(len(y_true), majority_label(y_train), dtype=np.int64)
    labels = list(labels) if labels is not None else sorted(set(y_true.tolist()) | set(pred.tolist()))
    return (
        float(f1_score(y_true, pred, average="macro", labels=labels, zero_division=0)),
        float(accuracy_score(y_true, pred)),
    )


def fold_macro_f1_is_defined(y_true: np.ndarray) -> bool:
    """Per-fold macro-F1 is only meaningful with ≥2 test samples of ≥2 classes.

    A single-sample (or single-class) fold makes macro-F1 a function of the
    label list, not of the model — those folds must be pooled instead.
    """
    y_true = np.asarray(y_true)
    return len(y_true) >= 2 and len(np.unique(y_true)) >= 2


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
    labels: Sequence[int] | None = None,
    y_train: np.ndarray | None = None,
) -> EvalResult:
    """Score one set of predictions.

    ``y_train`` is required for the majority baseline; without it the baseline
    fields are None (never computed from the test labels).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class = {}
    for i, lab in enumerate(labels):
        name = class_names[lab] if class_names and lab < len(class_names) else str(lab)
        per_class[name] = float(per[i])

    pr_auc = roc = None
    present = np.unique(y_true)
    if (
        y_proba is not None
        and y_proba.ndim == 2
        and y_proba.shape[1] >= 2
        and len(present) >= 2
    ):
        try:
            # Align proba columns to model classes if needed via labels arg;
            # only score classes actually present in y_true to avoid OvR warnings.
            if y_proba.shape[1] >= len(present):
                # Map: assume columns correspond to sorted training classes stored
                # as 0..C-1 matching `labels` when shapes equal.
                if y_proba.shape[1] == len(labels):
                    col_idx = [labels.index(int(c)) for c in present if int(c) in labels]
                    yt = y_true
                    yp = y_proba[:, [labels.index(int(c)) for c in present]]
                    # remap y to 0..K-1 for the subset
                    remap = {int(c): i for i, c in enumerate(present)}
                    yt_r = np.array([remap[int(v)] for v in yt])
                    if len(present) == 2:
                        roc = float(roc_auc_score(yt_r, yp[:, 1]))
                        pr_auc = float(average_precision_score(yt_r, yp[:, 1]))
                    else:
                        roc = float(
                            roc_auc_score(yt_r, yp, multi_class="ovr", average="macro")
                        )
                        aps = []
                        for j in range(len(present)):
                            yt_b = (yt_r == j).astype(int)
                            if yt_b.sum() == 0 or yt_b.sum() == len(yt_b):
                                continue
                            aps.append(average_precision_score(yt_b, yp[:, j]))
                        pr_auc = float(np.mean(aps)) if aps else None
        except ValueError:
            pass

    maj_f1 = maj_acc = None
    if y_train is not None and len(y_train):
        maj_f1, maj_acc = majority_baseline_metrics(y_true, y_train=y_train, labels=labels)
    return EvalResult(
        macro_f1=macro,
        accuracy=acc,
        confusion=cm,
        pr_auc=pr_auc,
        roc_auc=roc,
        per_class_f1=per_class,
        majority_baseline_f1=maj_f1,
        majority_baseline_acc=maj_acc,
    )


def pooled_evaluation(
    fold_true: Sequence[np.ndarray],
    fold_pred: Sequence[np.ndarray],
    labels: Sequence[int] | None = None,
    class_names: Sequence[str] | None = None,
    fold_baseline_pred: Sequence[np.ndarray] | None = None,
) -> EvalResult:
    """Leave-one-group-out headline: concatenate every fold's predictions, score once.

    Averaging per-fold macro-F1 is wrong for small folds (single-sample folds
    reduce it to accuracy). ``fold_baseline_pred`` are the per-fold
    train-majority predictions, pooled the same way.
    """
    y_true = np.concatenate([np.asarray(a) for a in fold_true])
    y_pred = np.concatenate([np.asarray(a) for a in fold_pred])
    ev = evaluate_predictions(y_true, y_pred, class_names=class_names, labels=labels)
    if fold_baseline_pred is not None:
        b = np.concatenate([np.asarray(a) for a in fold_baseline_pred])
        lab = list(labels) if labels is not None else sorted(set(y_true.tolist()) | set(b.tolist()))
        ev.majority_baseline_f1 = float(f1_score(y_true, b, average="macro", labels=lab, zero_division=0))
        ev.majority_baseline_acc = float(accuracy_score(y_true, b))
    ev.extras["n_folds"] = len(fold_true)
    ev.extras["n_test"] = int(len(y_true))
    return ev


def bootstrap_indices(fault_ids: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    """One fault-level bootstrap draw: sample faults WITH replacement, return row indices.

    A fault drawn k times contributes its rows k times — duplicates are kept.
    """
    fault_ids_a = np.asarray(fault_ids)
    faults = np.array(sorted(set(fault_ids_a.tolist())))
    rows_by_fault = {f: np.where(fault_ids_a == f)[0] for f in faults}
    drawn = rng.choice(faults, size=len(faults), replace=True)
    return np.concatenate([rows_by_fault[f] for f in drawn])


def _macro_f1_present(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int] | None = None) -> float:
    # classes present in this resample's truth (∩ labels, if given); absent classes are not scored.
    # Rows of other classes still count: e.g. a healthy window predicted OR is an OR false positive.
    present = np.unique(y_true)
    if labels is not None:
        present = np.intersect1d(present, np.asarray(labels))
    if len(present) == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0))


def bootstrap_ci_fault_level(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fault_ids: Sequence[str],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    labels: Sequence[int] | None = None,
) -> dict[str, float]:
    """Percentile CI for macro-F1, resampling whole faults (not windows).

    Each draw samples fault identities with replacement and keeps duplicates.
    Macro-F1 in each draw averages over the classes present in that draw,
    restricted to ``labels`` when given (e.g. faults only, excluding healthy).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = np.array([
        _macro_f1_present(y_true[idx], y_pred[idx], labels)
        for idx in (bootstrap_indices(fault_ids, rng) for _ in range(n_boot))
    ])
    scores = scores[np.isfinite(scores)]
    return {
        "point": _macro_f1_present(y_true, y_pred, labels),
        "mean": float(np.mean(scores)),
        "low": float(np.quantile(scores, alpha / 2)),
        "high": float(np.quantile(scores, 1 - alpha / 2)),
        "n_boot": int(n_boot),
        "n_faults": int(len(set(np.asarray(fault_ids).tolist()))),
        "note": "fault-level bootstrap; few faults → wide intervals",
    }


def paired_bootstrap_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    fault_ids: Sequence[str],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """CI for macro-F1(A) − macro-F1(B) with the SAME fault resample for both."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = bootstrap_indices(fault_ids, rng)
        diffs.append(_macro_f1_present(y_true[idx], pred_a[idx]) - _macro_f1_present(y_true[idx], pred_b[idx]))
    diffs = np.asarray(diffs)
    return {
        "point": _macro_f1_present(y_true, pred_a) - _macro_f1_present(y_true, pred_b),
        "low": float(np.quantile(diffs, alpha / 2)),
        "high": float(np.quantile(diffs, 1 - alpha / 2)),
        "p_le_0": float(np.mean(diffs <= 0)),
        "n_boot": int(n_boot),
    }


def mcnemar_exact(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, Any]:
    """Exact McNemar test (binomial) comparing two models on same items.

    Prefer fault-level aggregation before calling when windows leak.
    Power caveat: with n_fault=12 this test has essentially no power —
    report effect size (n01, n10) alongside p.
    """
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    n01 = int(np.sum(~a_correct & b_correct))  # A wrong, B right
    n10 = int(np.sum(a_correct & ~b_correct))  # A right, B wrong
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0, "note": "no discordant pairs"}
    # two-sided exact binomial under p=0.5
    p = float(2 * binom.cdf(min(n01, n10), n, 0.5))
    p = min(p, 1.0)
    return {
        "n01": n01,
        "n10": n10,
        "n_discordant": n,
        "p_value": p,
        "effect_n01_minus_n10": n01 - n10,
        "power_caveat": "Underpowered at n_fault≈12; interpret effect size, not p alone",
    }


def aggregate_fault_level_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fault_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Majority-vote predictions per fault_id for fault-level McNemar."""
    faults = sorted(set(fault_ids))
    yt, yp = [], []
    for f in faults:
        mask = np.asarray(fault_ids) == f
        # true label should be unique per fault
        yt.append(int(np.bincount(y_true[mask]).argmax()))
        yp.append(int(np.bincount(y_pred[mask]).argmax()))
    return np.array(yt), np.array(yp), faults
