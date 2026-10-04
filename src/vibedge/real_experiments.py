"""C1 / transfer / severity / deployment experiments on the downloaded datasets.

Every experiment goes through the same path:
  load recording → (optional deployment view) → windows → features (all groups,
  once; each feature set is a column subset) → make_splits → score + pool.

Feature tables are cached under results/cache/ keyed by dataset + settings, so
figures can be regenerated without re-extracting.

Speed: CWRU RPM fields, MaFaulDa filenames and Paderborn filename codes give
the measured shaft rate. Order / envelope features use it (oracle speed);
the deploy-time path would use vibedge.speed instead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from joblib import Parallel, delayed

from vibedge.deployment_view import to_deployment_view
from vibedge.evaluate import majority_label, pooled_evaluation
from vibedge.experiments import MAJORITY_LOO_NOTE, _fr_from_meta, _geo_from_meta, score_protocol
from vibedge.features import ALL_GROUPS, FEATURE_SET_GROUPS, extract_features
from vibedge.models import fit_model
from vibedge.splits import SplitIndices, check_no_group_overlap, freeze_manifest, make_splits
from vibedge.taxonomy import CLASSES, CLASS_TO_IDX, SampleMeta
from vibedge.windowing import make_windows

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = ROOT / "results" / "cache"

TRANSFER_CLASSES = ["healthy", "bearing_IR", "bearing_OR"]  # shared by CWRU, Paderborn, MFPT

# Speed modes for order/envelope features:
#   oracle    — measured shaft rate from metadata (RPM field / filename)
#   estimated — vibedge.speed.estimate_speed over the generic 10–60 Hz search
#   prior     — same estimator, searched only around the machine's rated speed
#               (±10%), which an installer knows from the motor nameplate
SPEED_MODES = ("oracle", "estimated", "prior")
GENERIC_SEARCH_HZ = (10.0, 60.0)
SPEED_PRIOR_HZ = {
    "cwru": (26.0, 33.0),       # 1730–1797 rpm
    "paderborn": (13.0, 28.0),  # 900 and 1500 rpm
    "mafaulda": (11.0, 61.0),   # 700–3600 rpm: the prior is no narrower than generic
    "mfpt": (22.0, 28.0),       # 25 Hz
}


@dataclass
class FeatureTable:
    """Window-level features for one dataset view. X[set] shares row order with meta."""

    dataset: str
    X: dict[str, np.ndarray]
    names: dict[str, list[str]]
    meta: list[dict[str, Any]]
    settings: dict[str, Any] = field(default_factory=dict)

    def col(self, key: str) -> list:
        return [m.get(key) for m in self.meta]

    @property
    def y(self) -> np.ndarray:
        return np.array([CLASS_TO_IDX[m["label"]] for m in self.meta], dtype=np.int64)

    def subset(self, mask: np.ndarray) -> "FeatureTable":
        idx = np.where(mask)[0]
        return FeatureTable(
            self.dataset,
            {k: v[idx] for k, v in self.X.items()},
            self.names,
            [self.meta[i] for i in idx],
            dict(self.settings),
        )


# ---------------------------------------------------------------------------
# per-recording worker: load → view → window → features
# ---------------------------------------------------------------------------

def _meta_row(m: SampleMeta) -> dict[str, Any]:
    ex = m.extras or {}
    return {
        "dataset": m.dataset,
        "recording_id": m.file_id,
        "fault_id": m.fault_id,
        "label": m.label,
        "severity": m.severity,
        "load": ex.get("operating_condition", m.load_hp),
        "fs": m.fs,
        "window_index": ex.get("window_index"),
        "artificial": ex.get("artificial"),
        "real_damage": ex.get("real_damage"),
        "subfault": ex.get("subfault"),
        "fr_true_hz": (m.rpm / 60.0) if m.rpm else ex.get("fr_hz"),
    }


def _features_for_recording(
    x: np.ndarray,
    meta: SampleMeta,
    window_s: float,
    overlap: float,
    max_windows: int | None,
    deployment: bool,
    slow_seconds: float,
    seed: int,
    speed: str = "oracle",
) -> list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, list[str]]]]:
    if deployment:
        x, fs = to_deployment_view(x, meta.fs, rng=np.random.default_rng(seed))
        meta.fs = fs
    out = []
    for xw, mw in make_windows([(x, meta)], window_s=window_s, overlap=overlap,
                               max_windows_per_recording=max_windows):
        if speed == "oracle":
            fr, search = _fr_from_meta(mw), GENERIC_SEARCH_HZ
        else:  # estimated from the window itself; metadata speed is never read
            fr = None
            search = SPEED_PRIOR_HZ[mw.dataset] if speed == "prior" else GENERIC_SEARCH_HZ
        r = extract_features(xw, mw.fs, geo=_geo_from_meta(mw), fr_hz=fr,
                             groups=ALL_GROUPS, slow_seconds=slow_seconds, search_hz=search)
        row = _meta_row(mw)
        row.update({"speed_mode": speed, "fr_used_hz": r.speed.fr_hz,
                    "speed_confidence": r.speed.confidence, "used_speed_features": r.used_speed_features})
        vecs, names = {}, {}
        for set_name, groups in FEATURE_SET_GROUPS.items():
            vecs[set_name], names[set_name] = r.subset(groups)
        out.append((row, vecs, names))
    return out


def _job(loader: Callable[[], Any], kw: dict[str, Any], seed: int):
    rec = loader()
    if rec is None:
        return []
    return _features_for_recording(rec.signal, rec.meta, seed=seed, **kw)


# ---------------------------------------------------------------------------
# dataset → list of zero-arg loaders (picklable via functools.partial)
# ---------------------------------------------------------------------------

def _loaders(dataset: str) -> list[Callable[[], Any]]:
    from functools import partial

    if dataset == "cwru":
        from vibedge.datasets.cwru import load_cwru_directory

        recs = load_cwru_directory(DATA / "cwru", target_fs=12000.0)
        return [partial(lambda r: r, r) for r in recs]
    if dataset == "mfpt":
        from vibedge.datasets.mfpt import load_mfpt

        return [partial(lambda r: r, r) for r in load_mfpt(DATA / "mfpt")]
    if dataset == "mafaulda":
        from vibedge.datasets.mafaulda import load_mafaulda_file, mafaulda_files

        root = DATA / "mafaulda"
        return [partial(load_mafaulda_file, root, p) for p in mafaulda_files(root)]
    if dataset == "paderborn":
        from vibedge.datasets.paderborn import load_paderborn_file

        return [partial(load_paderborn_file, p) for p in sorted((DATA / "paderborn").rglob("*.mat"))]
    raise ValueError(dataset)


# Window settings per dataset (pristine view). Recording lengths: CWRU ≈10 s,
# MaFaulDa 5 s, Paderborn 4 s, MFPT 3–6 s.
PRISTINE = {
    "cwru": dict(window_s=4.0, overlap=0.5, max_windows=None, slow_seconds=4.0),
    "mafaulda": dict(window_s=4.0, overlap=0.0, max_windows=1, slow_seconds=4.0),
    "paderborn": dict(window_s=3.9, overlap=0.0, max_windows=1, slow_seconds=3.9),
    "mfpt": dict(window_s=2.0, overlap=0.0, max_windows=None, slow_seconds=2.0),
}
# Common view for cross-dataset transfer: deployment sensor, 2 s windows.
DEPLOY = dict(window_s=2.0, overlap=0.0, max_windows=None, slow_seconds=2.0)


def build_table(dataset: str, deployment: bool = False, n_jobs: int = -1,
                refresh: bool = False, settings: dict | None = None, speed: str = "oracle") -> FeatureTable:
    if speed not in SPEED_MODES:
        raise ValueError(f"speed must be one of {SPEED_MODES}")
    kw = dict(settings or (DEPLOY if deployment else PRISTINE[dataset]))
    kw["deployment"] = deployment
    kw["speed"] = speed
    # the estimator's source is part of the key: changing speed.py invalidates the cache
    speed_src = hashlib.sha256((ROOT / "src" / "vibedge" / "speed.py").read_bytes()).hexdigest()[:12]
    key = hashlib.sha256(json.dumps({"ds": dataset, **kw, "v": 4, "speed_py": speed_src},
                                    sort_keys=True).encode()).hexdigest()[:10]
    cache = CACHE / f"{dataset}_{'deploy' if deployment else 'pristine'}_{speed}_{key}.npz"
    if cache.exists() and not refresh:
        z = np.load(cache, allow_pickle=True)
        return FeatureTable(dataset, dict(z["X"].item()), dict(z["names"].item()),
                            list(z["meta"]), dict(z["settings"].item()))

    loaders = _loaders(dataset)
    results = Parallel(n_jobs=n_jobs, batch_size=4)(
        delayed(_job)(ld, kw, i) for i, ld in enumerate(loaders)
    )
    rows = [r for res in results for r in res]
    if not rows:
        raise RuntimeError(f"no windows built for {dataset}")
    names = rows[0][2]
    X = {s: np.nan_to_num(np.vstack([r[1][s] for r in rows]), nan=0.0, posinf=0.0, neginf=0.0)
         for s in FEATURE_SET_GROUPS}
    meta = [r[0] for r in rows]
    table = FeatureTable(dataset, X, names, meta, {**kw, "n_recordings": len(loaders)})
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=np.array(X, dtype=object), names=np.array(names, dtype=object),
                        meta=np.array(meta, dtype=object), settings=np.array(table.settings, dtype=object))
    return table


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------

def _cols(t: FeatureTable) -> dict[str, list]:
    return {
        "recording_ids": t.col("recording_id"),
        "fault_ids": t.col("fault_id"),
        "labels": t.col("label"),
        "loads": t.col("load"),
        "severities": t.col("severity"),
        "datasets": t.col("dataset"),
    }


def _score_all_sets(t: FeatureTable, splits: Sequence[SplitIndices], protocol: Any,
                    model_name: str, n_boot: int = 1000,
                    excl_healthy: bool = False) -> list[dict[str, Any]]:
    """Score every feature set on these splits; pooled rows get a fault-level 95% CI,
    and the envelope_ratio row gets a paired CI for (envelope_ratio − time_only)."""
    from vibedge.evaluate import bootstrap_ci_fault_level, paired_bootstrap_diff

    y = t.y
    fids = np.asarray(t.col("fault_id"))
    labels = sorted(set(y.tolist()))
    rows, preds = [], {}
    for set_name in FEATURE_SET_GROUPS:
        folds, pooled = score_protocol(t.X[set_name], y, splits, labels, model_name=model_name,
                                       keep_predictions=True)
        idx, pred = pooled.pop("_test_idx"), pooled.pop("_y_pred")
        preds[set_name] = (idx, pred)
        if n_boot:
            pooled["ci95"] = bootstrap_ci_fault_level(y[idx], pred, fids[idx], n_boot=n_boot)
        if excl_healthy:
            # macro-F1 over fault classes only; healthy windows still count as false positives
            faults = [c for c in labels if CLASSES[c] != "healthy"]
            ci = bootstrap_ci_fault_level(y[idx], pred, fids[idx], n_boot=n_boot or 1000, labels=faults)
            pooled["macro_f1_excl_healthy"] = ci["point"]
            pooled["macro_f1_excl_healthy_ci95"] = ci
            pooled["macro_f1_excl_healthy_labels"] = [CLASSES[c] for c in faults]
        for r in folds + [pooled]:
            if r["fold"] != "pooled":
                r.pop("confusion", None)  # keep result files small; pooled keeps it
            rows.append({"feature_set": set_name, "protocol": protocol, **r})
    if n_boot and {"envelope_ratio", "time_only"} <= set(preds):
        (ia, pa), (ib, pb) = preds["envelope_ratio"], preds["time_only"]
        assert np.array_equal(ia, ib)
        diff = paired_bootstrap_diff(y[ia], pa, pb, fids[ia], n_boot=n_boot)
        for r in rows:
            if r["fold"] == "pooled" and r["feature_set"] == "envelope_ratio":
                r["diff_vs_time_only_ci95"] = diff
    return rows


def run_c1(t: FeatureTable, protocols: Sequence[int], model_name: str = "logreg",
           freeze: bool = True, extra_splits: dict[str, list[SplitIndices]] | None = None,
           n_boot: int = 1000, excl_healthy_protocols: Sequence[Any] = ()) -> dict[str, Any]:
    cols = _cols(t)
    rows, n_folds, skipped = [], {}, {}
    all_splits: dict[Any, list[SplitIndices]] = {}
    for p in protocols:
        try:
            sp = make_splits(p, **cols)
        except ValueError as e:
            skipped[p] = str(e)
            continue
        if sp:
            all_splits[p] = sp
    for name, sp in (extra_splits or {}).items():
        for s in sp:
            check_no_group_overlap(s, cols["recording_ids"], "recording")
        all_splits[name] = sp
    for p, sp in all_splits.items():
        if freeze:
            fdir = ROOT / "configs" / "frozen" / t.dataset
            for s in sp:
                freeze_manifest(s, cols["fault_ids"], cols["labels"], fdir,
                                name=f"protocol{p}_fold{s.fold}.json")
        n_folds[str(p)] = len(sp)
        rows += _score_all_sets(t, sp, p, model_name, n_boot=n_boot if p != 0 else 0,
                                excl_healthy=p in excl_healthy_protocols)
    return {
        "dataset": t.dataset,
        "model": model_name,
        "settings": t.settings,
        "n_windows": len(t.meta),
        "n_recordings": len(set(cols["recording_ids"])),
        "n_fault_identities": len(set(cols["fault_ids"])),
        "labels": [CLASSES[i] for i in sorted(set(t.y.tolist()))],
        "n_folds": n_folds,
        "skipped_protocols": skipped,
        "headline": "rows with fold == 'pooled'",
        "speed_mode": t.settings.get("speed", "oracle"),
        "notes": [MAJORITY_LOO_NOTE,
                  "ci95: fault-level bootstrap (1000 draws), not computed for P0 (random windows)."],
        "rows": rows,
    }


def speed_accuracy(t: FeatureTable, tol: float = 0.02) -> dict[str, Any]:
    """How close the speed used for features is to the measured speed."""
    true = np.array([m["fr_true_hz"] or np.nan for m in t.meta], dtype=float)
    used = np.array([m.get("fr_used_hz", np.nan) for m in t.meta], dtype=float)
    conf = np.array([m.get("speed_confidence", np.nan) for m in t.meta], dtype=float)
    ok = np.isfinite(true) & np.isfinite(used) & (true > 0)
    rel = np.abs(used[ok] - true[ok]) / true[ok]
    ratio = used[ok] / true[ok]
    return {
        "n_windows": int(ok.sum()),
        "within_2pct": float(np.mean(rel < tol)),
        "within_5pct": float(np.mean(rel < 0.05)),
        "median_rel_error": float(np.median(rel)),
        "locked_to_2x": float(np.mean(np.abs(ratio - 2) < 0.05)),
        "locked_to_half": float(np.mean(np.abs(ratio - 0.5) < 0.05)),
        "used_speed_features": float(np.mean([bool(m.get("used_speed_features")) for m in t.meta])),
        "median_confidence": float(np.nanmedian(conf)),
    }


def paderborn_artificial_to_real(t: FeatureTable) -> list[SplitIndices]:
    """Train on artificial damage, test on real damage; healthy bearings split by bearing.

    Healthy K001–K003 train, K004–K006 test (each bearing is one identity).
    """
    fid = np.asarray(t.col("fault_id"))
    art = np.asarray([bool(a) for a in t.col("artificial")])
    real = np.asarray([bool(r) for r in t.col("real_damage")])
    lab = np.asarray(t.col("label"))
    healthy_train = np.isin(fid, ["paderborn_K001", "paderborn_K002", "paderborn_K003"])
    healthy_test = (lab == "healthy") & ~healthy_train
    train = np.where(art | healthy_train)[0]
    test = np.where(real | healthy_test)[0]
    return [SplitIndices(3, train, test, fold=0, meta={"split": "artificial→real"})]


def mafaulda_severity(t: FeatureTable, model_name: str = "logreg") -> dict[str, Any]:
    """Held-out imbalance severity (healthy + imbalance only) — the honest imbalance protocol.

    interp: train {6,10,20,30,35} g, test {15,25} g.  extrap: train ≤30 g, test 35 g.
    Healthy recordings are dealt to train/test per recording (alternating).
    """
    from vibedge.datasets.mafaulda import IMBALANCE_TEST_SEVERITIES, IMBALANCE_TRAIN_SEVERITIES
    from vibedge.splits import protocol4_held_out_severity

    sub = t.subset(np.isin(np.asarray(t.col("label")), ["healthy", "imbalance"]))
    cols = _cols(sub)
    designs = {
        "interp": (IMBALANCE_TRAIN_SEVERITIES, IMBALANCE_TEST_SEVERITIES),
        "extrap": ([6.0, 10.0, 15.0, 20.0, 25.0, 30.0], [35.0]),
    }
    out: dict[str, Any] = {"dataset": "mafaulda", "classes": ["healthy", "imbalance"], "designs": {}}
    y = sub.y
    for name, (tr_s, te_s) in designs.items():
        sp = protocol4_held_out_severity(cols["severities"], cols["labels"], tr_s, te_s,
                                         recording_ids=cols["recording_ids"])
        check_no_group_overlap(sp, cols["recording_ids"], "recording")
        res = {"train_severities_g": tr_s, "test_severities_g": te_s,
               "n_train": int(len(sp.train_idx)), "n_test": int(len(sp.test_idx)), "by_set": {}}
        for set_name in FEATURE_SET_GROUPS:
            _, pooled = score_protocol(sub.X[set_name], y, [sp], sorted(set(y.tolist())), model_name=model_name)
            res["by_set"][set_name] = {k: pooled[k] for k in
                                       ("macro_f1", "accuracy", "majority_baseline_f1", "per_class_f1")}
        out["designs"][name] = res
    return out


def transfer_matrix(tables: dict[str, FeatureTable], feature_set: str = "envelope_ratio",
                    model_name: str = "logreg") -> dict[str, Any]:
    """Protocol 5: train on one dataset, test on another, classes {healthy, IR, OR}.

    Diagonal = within-dataset leave-one-fault-identity-out (pooled), or
    leave-one-recording-out where identities are too few (MFPT).
    """
    names = list(tables)
    subs = {n: t.subset(np.isin(np.asarray(t.col("label")), TRANSFER_CLASSES)) for n, t in tables.items()}
    labels = [CLASS_TO_IDX[c] for c in TRANSFER_CLASSES]
    M = np.full((len(names), len(names)), np.nan)
    B = np.full_like(M, np.nan)
    diag_protocol = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            ta, tb = subs[a], subs[b]
            if i == j:
                cols = _cols(ta)
                p = 3 if len(set(cols["fault_ids"])) > len(TRANSFER_CLASSES) else 1
                sp = make_splits(p, **cols)
                _, pooled = score_protocol(ta.X[feature_set], ta.y, sp, labels, model_name=model_name)
                diag_protocol[a] = p
            else:
                clf = fit_model(model_name, ta.X[feature_set], ta.y)
                pred = clf.predict(tb.X[feature_set])
                base = np.full(len(tb.y), majority_label(ta.y))
                ev = pooled_evaluation([tb.y], [pred], labels=labels, fold_baseline_pred=[base])
                pooled = {"macro_f1": ev.macro_f1, "majority_baseline_f1": ev.majority_baseline_f1}
            M[i, j] = pooled["macro_f1"]
            B[i, j] = pooled["majority_baseline_f1"]
    return {"datasets": names, "feature_set": feature_set, "classes": TRANSFER_CLASSES,
            "scores": M.tolist(), "majority_baseline": B.tolist(), "diagonal_protocol": diag_protocol,
            "view": "deployment (6 kHz LPF, 26.7 kSPS, MEMS noise), 2 s windows"}


# ---------------------------------------------------------------------------
# GBDT loss curves (log-loss / macro-F1 after every boosting stage)
# ---------------------------------------------------------------------------

def _fold_stage_curves(X, y, split, labels, n_estimators, max_depth, learning_rate, seed):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import log_loss

    tr, te = split.train_idx, split.test_idx
    clf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     learning_rate=learning_rate, random_state=seed)
    clf.fit(X[tr], y[tr])
    cols = [labels.index(int(c)) for c in clf.classes_]

    def full(p):  # model columns → all dataset labels (unseen classes get 0)
        out = np.zeros((p.shape[0], len(labels)))
        out[:, cols] = p
        return out

    train_loss = [float(log_loss(y[tr], full(p), labels=labels)) for p in clf.staged_predict_proba(X[tr])]
    test_proba = np.stack([full(p) for p in clf.staged_predict_proba(X[te])])  # (stages, n_test, C)
    unseen = bool(set(y[te].tolist()) - set(clf.classes_.tolist()))
    return train_loss, test_proba, te, unseen


def gbdt_loss_curves(
    t: FeatureTable,
    feature_sets: Sequence[str] = ("time_only", "envelope_ratio"),
    protocols: Sequence[int] = (0, 3),
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    n_jobs: int = -1,
    seed: int = 0,
) -> dict[str, Any]:
    """Train vs held-out log-loss and pooled macro-F1 per boosting stage.

    Held-out curves are POOLED: at each stage every fold's test predictions
    are concatenated and scored once. Folds whose test class never appears in
    training (e.g. CWRU healthy under P3) get probability 0 for the true class,
    so they are left out of the held-out LOSS (counted in
    ``n_folds_unseen_class``) but kept in the macro-F1 curve.
    """
    from sklearn.metrics import f1_score, log_loss

    cols = _cols(t)
    y = t.y
    labels = sorted(set(y.tolist()))
    out: dict[str, Any] = {"dataset": t.dataset, "model": "gbdt", "n_estimators": n_estimators,
                           "max_depth": max_depth, "learning_rate": learning_rate, "curves": {}}
    for fs in feature_sets:
        X = t.X[fs]
        for p in protocols:
            splits = make_splits(p, **cols)
            res = Parallel(n_jobs=n_jobs)(
                delayed(_fold_stage_curves)(X, y, sp, labels, n_estimators, max_depth, learning_rate, seed)
                for sp in splits
            )
            train_loss = np.mean([r[0] for r in res], axis=0)
            y_all = np.concatenate([y[r[2]] for r in res])
            P_all = np.concatenate([r[1] for r in res], axis=1)  # (stages, N, C)
            seen = np.concatenate([np.full(len(r[2]), not r[3]) for r in res])
            test_loss, test_f1 = [], []
            for s in range(n_estimators):
                Ps = P_all[s]
                test_loss.append(float(log_loss(y_all[seen], np.clip(Ps[seen], 1e-15, 1), labels=labels))
                                 if seen.any() else float("nan"))
                test_f1.append(float(f1_score(y_all, np.asarray(labels)[Ps.argmax(1)], average="macro",
                                              labels=labels, zero_division=0)))
            best = int(np.nanargmin(test_loss)) if seen.any() else None
            out["curves"][f"{fs}|P{p}"] = {
                "feature_set": fs, "protocol": p, "n_folds": len(splits),
                "n_folds_unseen_class": int(sum(r[3] for r in res)),
                "train_logloss": train_loss.tolist(), "heldout_logloss": test_loss,
                "heldout_macro_f1": test_f1,
                "best_stage_by_heldout_loss": None if best is None else best + 1,
                "final_heldout_macro_f1": test_f1[-1],
            }
    return out
