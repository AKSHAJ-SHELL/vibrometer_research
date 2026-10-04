"""Experiment runners — including the C1 feature-set × protocol hypothesis test.

Scoring rules (see DESIGN.md "Evaluation methods"):
  * the majority baseline is fit on each fold's TRAIN labels, never the test labels;
  * the headline number per (feature set, protocol) is the POOLED score — all
    folds' predictions concatenated and scored once (row with fold="pooled");
  * per-fold macro-F1 is None when the fold has < 2 test samples or < 2 classes;
  * macro-F1 averages over the classes present in the dataset, not all 6.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from vibedge.cost import cost_table
from vibedge.datasets.synthetic import generate_dataset
from vibedge.evaluate import (
    evaluate_predictions,
    fold_macro_f1_is_defined,
    majority_label,
    pooled_evaluation,
)
from vibedge.features import FEATURE_SET_GROUPS, extract_features
from vibedge.features.bearing import BearingGeometry
from vibedge.models import fit_model
from vibedge.splits import (
    SplitIndices,
    check_no_group_overlap,
    freeze_manifest,
    make_splits,
    protocol0_random,
    protocol3_all_folds,
    protocol4_held_out_severity,
    verify_manifest,
)
from vibedge.taxonomy import CLASSES, labels_to_indices
from vibedge.windowing import make_windows, meta_columns

__all__ = [
    "FEATURE_SET_GROUPS",
    "run_c1_synthetic",
    "run_c1_cwru",
    "run_c1_on_windows",
    "score_protocol",
    "run_severity_holdout_synthetic",
    "quick_macro_f1",
    "run_from_config",
]

MAJORITY_LOO_NOTE = (
    "Train-majority baseline under leave-one-group-out on balanced classes: "
    "holding out a group makes its class the training minority, so the "
    "baseline is wrong on every fold and its pooled macro-F1 is 0. This is "
    "expected behaviour, not a bug."
)


def _geo_from_meta(meta: Any) -> BearingGeometry | None:
    if not meta.geometry:
        return None
    return BearingGeometry(
        n=int(meta.geometry["n"]),
        d=float(meta.geometry["d"]),
        D=float(meta.geometry["D"]),
        phi_deg=float(meta.geometry.get("phi_deg", 0.0)),
    )


def _fr_from_meta(meta: Any) -> float | None:
    fr = meta.extras.get("fr_hz") if meta.extras else None
    if fr is None and meta.rpm:
        fr = meta.rpm / 60.0
    return fr


def _build_feature_matrix(
    pairs: Sequence[tuple[np.ndarray, Any]],
    groups: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[float | None]]:
    """pairs: list of (signal, SampleMeta). Only the requested groups are computed."""
    vectors = []
    labels = []
    fault_ids = []
    severities: list[float | None] = []
    names_ref: list[str] | None = None
    for x, meta in pairs:
        result = extract_features(
            x, meta.fs, geo=_geo_from_meta(meta), fr_hz=_fr_from_meta(meta), groups=groups
        )
        vec, names = result.subset(groups)
        if names_ref is None:
            names_ref = names
        if names != names_ref:
            d = dict(zip(names, vec))
            vec = np.array([d.get(n, 0.0) for n in names_ref], dtype=np.float64)
        vectors.append(vec)
        labels.append(meta.label)
        fault_ids.append(meta.fault_id)
        severities.append(meta.severity)
    X = np.nan_to_num(np.vstack(vectors), nan=0.0, posinf=0.0, neginf=0.0)
    y = labels_to_indices(labels)
    return X, y, fault_ids, labels, severities


def _eval_split(X, y, train_idx, test_idx, model_name="logreg", seed=0, labels=None):
    clf = fit_model(model_name, X[train_idx], y[train_idx], seed=seed)
    pred = clf.predict(X[test_idx])
    proba = clf.predict_proba(X[test_idx])
    labels = labels if labels is not None else sorted(set(y.tolist()))
    return evaluate_predictions(
        y[test_idx], pred, proba, class_names=CLASSES, labels=labels, y_train=y[train_idx]
    )


def score_protocol(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SplitIndices],
    labels: Sequence[int],
    model_name: str = "logreg",
    seed: int = 0,
    count_ids: tuple[Sequence[str], Sequence[str]] | None = None,
    keep_predictions: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score every fold, then pool. Returns (per-fold rows, pooled row).

    keep_predictions adds "_test_idx" / "_y_pred" arrays to the pooled row
    (for bootstrap CIs); callers must pop them before writing JSON.
    """
    labels = list(labels)
    fold_rows = []
    f_true, f_pred, f_base = [], [], []
    for split in splits:
        tr, te = split.train_idx, split.test_idx
        clf = fit_model(model_name, X[tr], y[tr], seed=seed)
        pred = np.asarray(clf.predict(X[te]), dtype=np.int64)
        base = np.full(len(te), majority_label(y[tr]), dtype=np.int64)
        ev = evaluate_predictions(y[te], pred, class_names=CLASSES, labels=labels, y_train=y[tr])
        defined = fold_macro_f1_is_defined(y[te])
        row = {
            "fold": split.fold,
            "n_test": int(len(te)),
            "macro_f1": ev.macro_f1 if defined else None,
            "accuracy": ev.accuracy,
            # kept numeric (tests/test_c1_experiment.py requires it); like macro_f1 it
            # is not interpretable on an undefined fold — use the pooled row
            "majority_baseline_f1": ev.majority_baseline_f1,
            "majority_baseline_acc": ev.majority_baseline_acc,
            "macro_f1_note": None if defined else "undefined: <2 test samples or <2 classes; see pooled row",
            "confusion": ev.confusion.tolist(),
            "meta": split.meta,
        }
        if count_ids is not None:
            row["counts"] = split.counts(*count_ids)
        fold_rows.append(row)
        f_true.append(y[te])
        f_pred.append(pred)
        f_base.append(base)
    pooled = pooled_evaluation(f_true, f_pred, labels=labels, class_names=CLASSES, fold_baseline_pred=f_base)
    pooled_row = {
        "fold": "pooled",
        "n_folds": len(splits),
        "n_test": pooled.extras["n_test"],
        "macro_f1": pooled.macro_f1,
        "accuracy": pooled.accuracy,
        "majority_baseline_f1": pooled.majority_baseline_f1,
        "majority_baseline_acc": pooled.majority_baseline_acc,
        "per_class_f1": pooled.per_class_f1,
        "confusion": pooled.confusion.tolist(),
        "labels": [CLASSES[i] for i in labels],
    }
    if keep_predictions:
        pooled_row["_test_idx"] = np.concatenate([s.test_idx for s in splits])
        pooled_row["_y_pred"] = np.concatenate(f_pred)
    return fold_rows, pooled_row


def _freeze_or_verify(splits, fault_ids, labels_str, freeze_dir, require_frozen):
    if freeze_dir is None:
        return
    for split in splits:
        if require_frozen:
            verify_manifest(split, fault_ids, labels_str, freeze_dir, require=True)
        else:
            freeze_manifest(split, fault_ids, labels_str, freeze_dir)


def run_c1_on_windows(
    windows: Sequence[tuple[np.ndarray, Any]],
    protocols: Sequence[int],
    freeze_dir: str | Path | None = None,
    require_frozen: bool = False,
    model_name: str = "logreg",
    feature_sets: dict[str, list[str]] | None = None,
    dataset_name: str = "",
    cost_repeats: int = 3,
) -> dict[str, Any]:
    """C1 on arbitrary windowed data, splits from make_splits (leakage-checked)."""
    feature_sets = feature_sets or FEATURE_SET_GROUPS
    cols = meta_columns(windows)
    y_all = labels_to_indices(cols["labels"])
    label_universe = sorted(set(y_all.tolist()))

    splits_by_protocol: dict[int, list[SplitIndices]] = {}
    skipped: dict[int, str] = {}
    for p in protocols:
        try:
            sp = make_splits(p, **cols)
        except ValueError as e:
            skipped[p] = str(e)
            continue
        if not sp:
            skipped[p] = "no feasible folds"
            continue
        _freeze_or_verify(sp, cols["fault_ids"], cols["labels"], freeze_dir, require_frozen)
        splits_by_protocol[p] = sp

    rows = []
    X_by_set = {}
    for fs_name, groups in feature_sets.items():
        X, y, *_ = _build_feature_matrix(windows, groups)
        X_by_set[fs_name] = X
        for p, sp in splits_by_protocol.items():
            folds, pooled = score_protocol(
                X, y, sp, label_universe, model_name=model_name,
                count_ids=(cols["fault_ids"], cols["labels"]),
            )
            for r in folds + [pooled]:
                rows.append({"feature_set": fs_name, "protocol": p, **r})

    x0, m0 = windows[0]
    cost = cost_table(
        x0, m0.fs, _fr_from_meta(m0), X_by_set=X_by_set, y=y_all,
        geo=_geo_from_meta(m0), feature_sets=feature_sets, repeats=cost_repeats,
    )
    return {
        "dataset": dataset_name,
        "hypothesis": "envelope_ratio macro-F1 degrades less than time_only as protocol rigor increases",
        "model": model_name,
        "n_windows": len(windows),
        "n_recordings": len(set(cols["recording_ids"])),
        "n_fault_identities": len(set(cols["fault_ids"])),
        "n_folds": {p: len(sp) for p, sp in splits_by_protocol.items()},
        "skipped_protocols": skipped,
        "labels": [CLASSES[i] for i in label_universe],
        "headline": "rows with fold == 'pooled'",
        "notes": [MAJORITY_LOO_NOTE],
        "rows": rows,
        "cost": cost,
    }


def run_c1_synthetic(
    n_per_class: int = 4,
    freeze_dir: str | Path = "configs/frozen",
    require_frozen: bool = False,
    protocols: list[int] | None = None,
    model_name: str = "logreg",
) -> dict[str, Any]:
    """Core hypothesis on synthetic data (one sample per recording).

    Protocol 4 keeps the original pre-registered single split
    (train severities 0.5/1.0/2.0, test 1.5) so frozen manifests still match.
    """
    protocols = protocols or [0, 3, 4]
    rows_data = generate_dataset(n_per_class=n_per_class, duration=2.0, base_seed=42)
    pairs = [(x, m) for x, _, m in rows_data]
    recording_ids = [m.file_id for _, m in pairs]

    rows = []
    X_by_set = {}
    y_ref = None
    for fs_name, groups in FEATURE_SET_GROUPS.items():
        X, y, fault_ids, labels, severities = _build_feature_matrix(pairs, groups)
        X_by_set[fs_name] = X
        y_ref = y
        label_universe = sorted(set(y.tolist()))

        for protocol in protocols:
            if protocol == 0:
                splits = [protocol0_random(len(y), seed=0)]
            elif protocol == 3:
                splits = protocol3_all_folds(fault_ids)
            elif protocol == 4:
                try:
                    splits = [
                        protocol4_held_out_severity(
                            severities, labels, [0.5, 1.0, 2.0], [1.5], recording_ids=recording_ids
                        )
                    ]
                except ValueError:
                    continue
            else:
                continue
            if protocol >= 1:
                for sp in splits:
                    check_no_group_overlap(sp, recording_ids, "recording")
            _freeze_or_verify(splits, fault_ids, labels, freeze_dir, require_frozen)

            folds, pooled = score_protocol(
                X, y, splits, label_universe, model_name=model_name, count_ids=(fault_ids, labels)
            )
            for r in folds + [pooled]:
                rows.append({"feature_set": fs_name, "protocol": protocol, **r})

    x0, m0 = pairs[0]
    cost = cost_table(x0, m0.fs, _fr_from_meta(m0), X_by_set=X_by_set, y=y_ref,
                      geo=_geo_from_meta(m0), repeats=3)
    return {
        "dataset": "synthetic",
        "hypothesis": (
            "envelope_ratio macro-F1 degrades less than time_only from protocol 0→3/4"
        ),
        "model": model_name,
        "headline": "rows with fold == 'pooled'",
        "notes": [MAJORITY_LOO_NOTE],
        "rows": rows,
        "cost": cost,
    }


def run_c1_cwru(
    root: str | Path = "data/cwru",
    protocols: Sequence[int] = (0, 1, 2, 3, 4),
    window_s: float = 4.0,
    overlap: float = 0.5,
    target_fs: float = 12000.0,
    freeze_dir: str | Path | None = "configs/frozen/cwru",
    require_frozen: bool = False,
    model_name: str = "logreg",
) -> dict[str, Any]:
    """C1 on the CWRU 12k drive-end set + normal baseline (resampled to 12 kHz)."""
    from vibedge.datasets.cwru import load_cwru_directory

    recs = load_cwru_directory(root, target_fs=target_fs)
    if not recs:
        raise FileNotFoundError(f"no CWRU files under {root}; run scripts/download_data.py --dataset cwru")
    windows = make_windows([(r.signal, r.meta) for r in recs], window_s=window_s, overlap=overlap)
    out = run_c1_on_windows(
        windows, protocols, freeze_dir=freeze_dir, require_frozen=require_frozen,
        model_name=model_name, dataset_name="cwru_12k_DE+normal",
    )
    out["window_s"] = window_s
    out["overlap"] = overlap
    out["fs"] = target_fs
    out["files"] = sorted(r.meta.file_id for r in recs)
    out["notes"].append(
        "Healthy is a single fault identity (one bearing, 97.mat + 100.mat after KNOWN_BAD); "
        "under protocol 3 its fold trains with no healthy data."
    )
    return out


def run_severity_holdout_synthetic() -> dict[str, Any]:
    rows_data = generate_dataset(
        n_per_class=4,
        severities=[0.5, 1.0, 1.5, 2.0],
        duration=2.0,
        base_seed=3,
    )
    # keep imbalance + healthy only for a clean dose-response protocol demo
    pairs = [(x, m) for x, _, m in rows_data if m.label in ("healthy", "imbalance")]
    recording_ids = [m.file_id for _, m in pairs]
    X, y, fault_ids, labels, severities = _build_feature_matrix(
        pairs, FEATURE_SET_GROUPS["full"]
    )
    split_i = protocol4_held_out_severity(
        severities, labels, [0.5, 1.0, 2.0], [1.5], recording_ids=recording_ids
    )
    ev_i = _eval_split(X, y, split_i.train_idx, split_i.test_idx)
    # extrapolation: train low, test high
    try:
        split_e = protocol4_held_out_severity(
            severities, labels, [0.5, 1.0], [2.0], recording_ids=recording_ids
        )
        ev_e = _eval_split(X, y, split_e.train_idx, split_e.test_idx)
        extrap = ev_e.macro_f1
    except ValueError:
        extrap = float("nan")
    return {
        "interp_f1": ev_i.macro_f1,
        "extrap_f1": extrap,
        "majority_f1": ev_i.majority_baseline_f1,
    }


def quick_macro_f1(pairs: Sequence[tuple[np.ndarray, Any]], seed: int = 0) -> float:
    X, y, fault_ids, labels, _ = _build_feature_matrix(
        pairs, FEATURE_SET_GROUPS["envelope_ratio"]
    )
    split = protocol0_random(len(y), seed=seed)
    ev = _eval_split(X, y, split.train_idx, split.test_idx, seed=seed)
    return ev.macro_f1


def pooled_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in result["rows"] if r.get("fold") == "pooled"]


def run_from_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    freeze_dir = cfg.get("splits", {}).get("freeze_dir", "configs/frozen")
    # For default/CI path we freeze then score (first-run friendly)
    result = run_c1_synthetic(
        freeze_dir=freeze_dir,
        require_frozen=False,
        protocols=cfg.get("splits", {}).get("protocols", [0, 3, 4]),
    )
    out = Path(cfg.get("paths", {}).get("results_root", "results"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "c1_results.json").write_text(json.dumps(result, indent=2, default=_json_default))
    (out / "cost_table.json").write_text(json.dumps(result["cost"], indent=2, default=_json_default))
    return result


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))
