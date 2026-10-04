"""Spec §2 — Leakage. The paper's core claim depends on these.

Targets: vibedge.splits, vibedge.windowing, vibedge.datasets.cwru
"""

import numpy as np
import pytest
from draft_paths import REAL_CWRU, real_data
from scipy.io import savemat

from vibedge.datasets.cwru import KNOWN_BAD, load_cwru_directory
from vibedge.splits import LeakageError, SplitIndices, check_no_group_overlap, make_splits
from vibedge.taxonomy import SampleMeta
from vibedge.windowing import make_windows, meta_columns


def _three_recordings_ten_windows():
    """r0/r1/r2 × 10 windows; metadata makes protocols 1–5 all feasible."""
    fs = 1000.0
    spec = [
        ("r0", "healthy", "id_h", 0, None, "ds_a"),
        ("r1", "bearing_IR", "id_ir7", 1, 0.007, "ds_a"),
        ("r2", "bearing_IR", "id_ir14", 2, 0.014, "ds_b"),
    ]
    recs = []
    for k, (rid, label, fid, load, sev, ds) in enumerate(spec):
        m = SampleMeta(dataset=ds, file_id=rid, fault_id=fid, label=label, severity=sev, load_hp=load, fs=fs)
        recs.append((np.random.default_rng(k).standard_normal(10_000), m))
    windows = make_windows(recs, window_s=1.0)
    assert len(windows) == 30
    return windows


@pytest.mark.parametrize("protocol", [1, 2, 3, 4, 5])
def test_no_recording_overlap(protocol):
    # 2.1
    cols = meta_columns(_three_recordings_ten_windows())
    rec = np.asarray(cols["recording_ids"])
    splits = make_splits(protocol, **cols)
    assert splits, f"protocol {protocol} produced no folds"
    for sp in splits:
        shared = set(rec[sp.train_idx]) & set(rec[sp.test_idx])
        assert shared == set(), f"protocol {protocol} fold {sp.fold}: recordings on both sides: {sorted(shared)}"


def test_leaky_split_fails_loudly_with_offending_ids():
    # 2.1 negative case — r1 on both sides
    rec = ["r0"] * 10 + ["r1"] * 10 + ["r2"] * 10
    leaky = SplitIndices(1, train_idx=np.arange(0, 15), test_idx=np.arange(15, 30))
    with pytest.raises(LeakageError, match="r1"):
        check_no_group_overlap(leaky, rec)


def _write_fault_named_cwru(root):
    rng = np.random.default_rng(0)
    for stem in ("IR007", "B007", "OR007@6", "IR014", "Normal"):
        for load in range(4):
            savemat(root / f"{stem}_{load}.mat", {"X000_DE_time": rng.standard_normal(4000)})


def test_cwru_same_notch_across_loads_is_one_fault_identity(tmp_path):
    # 2.2
    _write_fault_named_cwru(tmp_path)
    ids = {r.meta.file_id: r.meta.fault_id for r in load_cwru_directory(tmp_path, subsets=None)}
    for load in range(4):
        assert ids[f"IR007_{load}.mat"] == "cwru_IR_0.007"


def test_cwru_protocol3_puts_all_loads_of_a_notch_in_one_test_fold(tmp_path):
    # 2.2 — the leakage the published 99% numbers hide
    _write_fault_named_cwru(tmp_path)
    recs = load_cwru_directory(tmp_path, subsets=None)
    windows = make_windows([(r.signal, r.meta) for r in recs], window_s=1000 / 48000.0)
    cols = meta_columns(windows)
    files = np.asarray(cols["recording_ids"])
    fids = np.asarray(cols["fault_ids"])
    loads = np.asarray(cols["loads"], dtype=object)
    splits = make_splits(3, **cols)

    ir007 = {f"IR007_{load}.mat" for load in range(4)}
    holding = [sp for sp in splits if ir007 & set(files[sp.test_idx])]
    assert len(holding) == 1, "IR007 recordings spread across several test folds"
    assert ir007 <= set(files[holding[0].test_idx])

    for sp in splits:
        for notch in set(fids[sp.test_idx]):
            tr = set(loads[sp.train_idx][fids[sp.train_idx] == notch])
            te = set(loads[sp.test_idx][fids[sp.test_idx] == notch])
            assert not (tr and te), f"fold {sp.fold}: {notch} loads {sorted(tr)} in train, {sorted(te)} in test"


def test_protocol3_folds_equal_fault_identities_not_files(tmp_path):
    # 2.3
    _write_fault_named_cwru(tmp_path)
    recs = load_cwru_directory(tmp_path, subsets=None)
    cols = meta_columns([(r.signal, r.meta) for r in recs])
    splits = make_splits(3, **cols)
    assert len(splits) == len(set(cols["fault_ids"])) == 5
    assert len(splits) < len(set(cols["recording_ids"])) == 20


@real_data
def test_protocol3_on_real_cwru_groups_by_identity():
    # 2.3 real data — expected today: 14 identities, 52 files
    recs = load_cwru_directory(REAL_CWRU, target_fs=12000.0)
    cols = meta_columns([(r.signal, r.meta) for r in recs])
    n_ids = len(set(cols["fault_ids"]))
    n_files = len(set(cols["recording_ids"]))
    assert len(make_splits(3, **cols)) == n_ids
    assert n_ids < n_files, "n_folds == n_files on real CWRU: identity grouping is not happening"


def test_known_bad_files_are_excluded(tmp_path):
    # 2.4
    assert "191.mat" in KNOWN_BAD
    rng = np.random.default_rng(1)
    for num in ("191", "105", "106", "97", "118"):
        n = int(num)
        savemat(tmp_path / f"{num}.mat", {f"X{n:03d}_DE_time": rng.standard_normal(4000), f"X{n:03d}RPM": 1797})
    recs = load_cwru_directory(tmp_path, subsets=None)
    names = {r.meta.file_id for r in recs}
    assert "191.mat" not in names
    assert {"105.mat", "106.mat", "97.mat", "118.mat"} <= names
    cols = meta_columns([(r.signal, r.meta) for r in recs])
    for protocol in (1, 3):
        for sp in make_splits(protocol, **cols):
            used = {cols["recording_ids"][i] for i in np.concatenate([sp.train_idx, sp.test_idx])}
            assert not used & set(KNOWN_BAD)


@real_data
def test_known_bad_absent_from_real_cwru():
    # 2.4 real data — 191.mat is on disk but never loaded
    assert "191.mat" in {p.name for p in REAL_CWRU.rglob("*.mat")}
    names = {r.meta.file_id for r in load_cwru_directory(REAL_CWRU, subsets=None)}
    assert not names & set(KNOWN_BAD)
