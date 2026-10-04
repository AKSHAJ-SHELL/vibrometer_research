"""Leakage ladder (protocols 0–5) with frozen hashed manifests.

Protocol 0 — random window split          INVALID (never report as result)
Protocol 1 — leave-one-file-out           weak
Protocol 2 — cross-load                   still leaky on CWRU
Protocol 3 — fault-identity split         minimum acceptable
Protocol 4 — held-out severity            closest to deployment
Protocol 5 — cross-dataset                most honest

Pre-registration: freeze_manifest() writes sha256 to configs/frozen/.
The runner refuses to score a test set whose manifest hash isn't frozen
when require_frozen_for_test is True.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vibedge.taxonomy import PROTOCOL_FEASIBILITY


@dataclass
class SplitIndices:
    protocol: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    fold: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def counts(self, fault_ids: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
        def uniq(idxs: np.ndarray) -> dict[str, int]:
            f = {fault_ids[i] for i in idxs}
            labs = {labels[i] for i in idxs}
            return {
                "n_windows": int(len(idxs)),
                "n_faults": len(f),
                "n_labels": len(labs),
                "faults": sorted(f),
                "labels": sorted(labs),
            }

        return {"train": uniq(self.train_idx), "test": uniq(self.test_idx)}


def _check_no_overlap(train: np.ndarray, test: np.ndarray) -> None:
    inter = np.intersect1d(train, test)
    if len(inter):
        raise RuntimeError(f"Split leakage: {len(inter)} indices in both train and test")


class LeakageError(RuntimeError):
    """A group (recording / fault identity) appears on both sides of a split."""


def check_no_group_overlap(
    split: SplitIndices, groups: Sequence[str], what: str = "recording"
) -> None:
    """Raise LeakageError naming every group id present in both train and test."""
    g = np.asarray(groups, dtype=object)
    shared = sorted(set(g[split.train_idx].tolist()) & set(g[split.test_idx].tolist()))
    if shared:
        raise LeakageError(
            f"protocol {split.protocol} fold {split.fold}: {len(shared)} {what} id(s) "
            f"on both sides of the split: {shared}"
        )


def protocol0_random(
    n: int, test_frac: float = 0.2, seed: int = 0
) -> SplitIndices:
    """INVALID — retained only to reproduce the collapse figure."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    test = np.sort(idx[:n_test])
    train = np.sort(idx[n_test:])
    _check_no_overlap(train, test)
    return SplitIndices(0, train, test, meta={"warning": "INVALID_RANDOM_WINDOW_SPLIT"})


def protocol1_leave_one_file(
    file_ids: Sequence[str], fold: int = 0
) -> SplitIndices:
    files = sorted(set(file_ids))
    if not files:
        raise ValueError("No files")
    held = files[fold % len(files)]
    file_ids_a = np.asarray(file_ids)
    test = np.where(file_ids_a == held)[0]
    train = np.where(file_ids_a != held)[0]
    _check_no_overlap(train, test)
    return SplitIndices(1, train, test, fold=fold, meta={"held_file": held})


def protocol2_cross_load(
    loads: Sequence[float | None],
    train_loads: Sequence[float],
    test_loads: Sequence[float],
) -> SplitIndices:
    loads_a = np.asarray(loads, dtype=object)
    train_set = set(train_loads)
    test_set = set(test_loads)
    train = np.array([i for i, L in enumerate(loads_a) if L in train_set], dtype=np.int64)
    test = np.array([i for i, L in enumerate(loads_a) if L in test_set], dtype=np.int64)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Empty train or test in cross-load split")
    _check_no_overlap(train, test)
    return SplitIndices(
        2, train, test, meta={"train_loads": list(train_loads), "test_loads": list(test_loads)}
    )


def protocol3_fault_identity(
    fault_ids: Sequence[str], fold: int = 0
) -> list[SplitIndices]:
    """Leave-one-fault-identity-out. fold >= 0 returns that fold only; fold < 0 all folds."""
    faults = sorted(set(fault_ids))
    fault_ids_a = np.asarray(fault_ids)
    splits = []
    for i, held in enumerate(faults):
        test = np.where(fault_ids_a == held)[0]
        train = np.where(fault_ids_a != held)[0]
        _check_no_overlap(train, test)
        splits.append(
            SplitIndices(3, train, test, fold=i, meta={"held_fault": held, "n_faults": len(faults)})
        )
    if 0 <= fold < len(splits):
        return [splits[fold]]
    return splits


def protocol3_all_folds(fault_ids: Sequence[str]) -> list[SplitIndices]:
    return protocol3_fault_identity(fault_ids, fold=-1)


def protocol4_held_out_severity(
    severities: Sequence[float | None],
    labels: Sequence[str],
    train_severities: Sequence[float],
    test_severities: Sequence[float],
    always_include_labels: Sequence[str] = ("healthy",),
    recording_ids: Sequence[str] | None = None,
    healthy_test_slot: int = 1,
    healthy_n_slots: int = 2,
) -> SplitIndices:
    """Held-out severity — the honest imbalance protocol on MaFaulDa.

    Healthy data has no severity, so healthy RECORDINGS are dealt round-robin:
    recording k goes to test iff k % healthy_n_slots == healthy_test_slot.
    Assignment is per recording, never per window — splitting one recording's
    windows across train and test is leakage. Without ``recording_ids`` each
    index is treated as its own recording.
    """
    train_s = set(float(s) for s in train_severities)
    test_s = set(float(s) for s in test_severities)
    rec = list(recording_ids) if recording_ids is not None else [str(i) for i in range(len(labels))]
    healthy_recs = sorted({rec[i] for i, lab in enumerate(labels) if lab in always_include_labels})
    healthy_test = {r for k, r in enumerate(healthy_recs) if k % healthy_n_slots == healthy_test_slot}
    train_idx = []
    test_idx = []
    for i, (sev, lab) in enumerate(zip(severities, labels)):
        if lab in always_include_labels:
            (test_idx if rec[i] in healthy_test else train_idx).append(i)
            continue
        if sev is None:
            continue
        s = float(sev)
        if s in train_s:
            train_idx.append(i)
        elif s in test_s:
            test_idx.append(i)
    train = np.array(sorted(train_idx), dtype=np.int64)
    test = np.array(sorted(test_idx), dtype=np.int64)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Empty train/test in severity split")
    _check_no_overlap(train, test)
    return SplitIndices(
        4,
        train,
        test,
        meta={
            "train_severities": list(train_severities),
            "test_severities": list(test_severities),
        },
    )


def protocol5_cross_dataset(
    datasets: Sequence[str],
    train_dataset: str,
    test_dataset: str,
) -> SplitIndices:
    ds = np.asarray(datasets)
    train = np.where(ds == train_dataset)[0]
    test = np.where(ds == test_dataset)[0]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"Empty split for {train_dataset}→{test_dataset}")
    _check_no_overlap(train, test)
    return SplitIndices(
        5,
        train,
        test,
        meta={"train_dataset": train_dataset, "test_dataset": test_dataset},
    )


def make_splits(
    protocol: int,
    *,
    recording_ids: Sequence[str],
    fault_ids: Sequence[str],
    labels: Sequence[str],
    loads: Sequence[float | None] | None = None,
    severities: Sequence[float | None] | None = None,
    datasets: Sequence[str] | None = None,
    healthy_labels: Sequence[str] = ("healthy",),
    seed: int = 0,
) -> list[SplitIndices]:
    """All folds of one protocol, over window-level metadata.

    One entry per window; ``recording_ids`` names the recording (file) each
    window was cut from. Every fold of protocol >= 1 is checked so that no
    recording is on both sides (LeakageError otherwise); protocol 3 also
    checks fault identity.

      1 leave-one-recording-out     2 leave-one-load-out
      3 leave-one-fault-identity-out 4 leave-one-severity-out (healthy dealt by recording)
      5 leave-one-dataset-out
    """
    n = len(recording_ids)
    for name, arr in [("fault_ids", fault_ids), ("labels", labels), ("loads", loads),
                      ("severities", severities), ("datasets", datasets)]:
        if arr is not None and len(arr) != n:
            raise ValueError(f"{name} has {len(arr)} entries, recording_ids has {n}")

    def leave_one_value_out(values: Sequence, proto: int, key: str) -> list[SplitIndices]:
        vals = np.asarray([None if v is None else v for v in values], dtype=object)
        present = sorted({v for v in vals.tolist() if v is not None}, key=str)
        out = []
        for i, held in enumerate(present):
            test = np.array([j for j in range(n) if vals[j] == held], dtype=np.int64)
            train = np.array([j for j in range(n) if vals[j] is not None and vals[j] != held], dtype=np.int64)
            if len(train) and len(test):
                out.append(SplitIndices(proto, train, test, fold=len(out), meta={key: held}))
        return out

    if protocol == 0:
        splits = [protocol0_random(n, seed=seed)]
    elif protocol == 1:
        splits = leave_one_value_out(recording_ids, 1, "held_file")
    elif protocol == 2:
        if loads is None:
            raise ValueError("protocol 2 needs loads")
        splits = leave_one_value_out(loads, 2, "held_load")
    elif protocol == 3:
        splits = protocol3_all_folds(fault_ids)
    elif protocol == 4:
        if severities is None:
            raise ValueError("protocol 4 needs severities")
        sev_levels = sorted({float(s) for s, lab in zip(severities, labels)
                             if s is not None and lab not in healthy_labels})
        splits = []
        for k, held in enumerate(sev_levels):
            try:
                sp = protocol4_held_out_severity(
                    severities, labels, [s for s in sev_levels if s != held], [held],
                    always_include_labels=healthy_labels, recording_ids=recording_ids,
                    healthy_test_slot=k % max(len(sev_levels), 1),
                    healthy_n_slots=max(len(sev_levels), 1),
                )
            except ValueError:
                continue
            sp.fold = len(splits)
            splits.append(sp)
    elif protocol == 5:
        if datasets is None:
            raise ValueError("protocol 5 needs datasets")
        splits = leave_one_value_out(datasets, 5, "held_dataset")
    else:
        raise ValueError(f"unknown protocol {protocol}")

    if protocol >= 1:
        for sp in splits:
            check_no_group_overlap(sp, recording_ids, "recording")
            if protocol == 3:
                check_no_group_overlap(sp, fault_ids, "fault identity")
    return splits


# ---------------------------------------------------------------------------
# Manifest freeze / verify (pre-registration mechanism)
# ---------------------------------------------------------------------------

def manifest_payload(split: SplitIndices, fault_ids: Sequence[str], labels: Sequence[str]) -> dict:
    return {
        "protocol": split.protocol,
        "fold": split.fold,
        "train_idx": split.train_idx.tolist(),
        "test_idx": split.test_idx.tolist(),
        "meta": split.meta,
        "counts": split.counts(fault_ids, labels),
    }


def manifest_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def freeze_manifest(
    split: SplitIndices,
    fault_ids: Sequence[str],
    labels: Sequence[str],
    freeze_dir: str | Path,
    name: str | None = None,
) -> Path:
    freeze_dir = Path(freeze_dir)
    freeze_dir.mkdir(parents=True, exist_ok=True)
    payload = manifest_payload(split, fault_ids, labels)
    h = manifest_hash(payload)
    payload["sha256"] = h
    name = name or f"protocol{split.protocol}_fold{split.fold}_{h[:12]}.json"
    path = freeze_dir / name
    path.write_text(json.dumps(payload, indent=2))
    return path


def verify_manifest(
    split: SplitIndices,
    fault_ids: Sequence[str],
    labels: Sequence[str],
    freeze_dir: str | Path,
    require: bool = True,
) -> bool:
    """Return True if a frozen manifest matches this split.

    If require=True and no match, raise RuntimeError (refuse to score).
    """
    freeze_dir = Path(freeze_dir)
    payload = manifest_payload(split, fault_ids, labels)
    h = manifest_hash(payload)
    if not freeze_dir.exists():
        if require:
            raise RuntimeError(
                f"No freeze dir {freeze_dir}; refuse to score unfrozen split {h[:12]}"
            )
        return False
    for p in freeze_dir.glob("*.json"):
        data = json.loads(p.read_text())
        if data.get("sha256") == h:
            return True
        # also accept if recomputed hash of stored idxs matches
        stored = {
            "protocol": data["protocol"],
            "fold": data["fold"],
            "train_idx": data["train_idx"],
            "test_idx": data["test_idx"],
            "meta": data.get("meta", {}),
            "counts": data.get("counts", {}),
        }
        if manifest_hash(stored) == h or data.get("sha256") == manifest_hash(stored):
            if (
                data["train_idx"] == payload["train_idx"]
                and data["test_idx"] == payload["test_idx"]
            ):
                return True
    if require:
        raise RuntimeError(
            f"Split hash {h[:12]} not frozen in {freeze_dir}; refuse to score. "
            f"Run freeze_manifest first (pre-registration)."
        )
    return False


def feasibility(dataset: str, protocol: int) -> str:
    return PROTOCOL_FEASIBILITY.get(dataset, {}).get(protocol, "unknown")
