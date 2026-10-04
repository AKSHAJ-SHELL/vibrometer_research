"""CWRU Bearing Data Center loader with KNOWN_BAD exclusions (Smith & Randall 2015).

Downloaded files are numbered (``105.mat``); fault type / size / load / OR
position come from ``cwru_catalog.CATALOG``, not the filename. Fault-named
files (``IR007_0.mat``) are also parsed, for tests and renamed mirrors.

Sampling rate of normal-baseline files is undocumented; record lengths match
48 kHz fault files — inference, not documentation. Flagged on resample.

Fault identity ignores load AND sampling rate: the 12k and 48k drive-end sets
reuse the same seeded bearings, so ``IR007`` at any load or rate is one
physical fault. Over-grouping is the conservative direction for leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from vibedge.datasets.cwru_catalog import CATALOG, SUBSET_FS
from vibedge.features.bearing import CWRU_6205_DE, CWRU_6203_FE
from vibedge.taxonomy import SampleMeta, map_cwru_fault_type

# Files diagnosed bad by Smith & Randall (2015) + packaging bugs we confirmed.
KNOWN_BAD: dict[str, str] = {
    "98.mat": "no RPM variable",
    "99.mat": "contains both X098_* and X099_* arrays — packaging bug",
    "3001.mat": "variable named X056_DE_time; DE only, no FE, no RPM",
    "109.mat": "243938 samples — half the length of siblings",
    "177.mat": "corrupted by dynamometer-controller electrical noise",
    "283.mat": "corrupted by dynamometer-controller electrical noise",
    "189.mat": "DE and FE channels identical (acquisition fault)",
    "201.mat": "DE and FE channels identical (acquisition fault)",
    "213.mat": "DE and FE channels identical (acquisition fault)",
    "226.mat": "DE and FE channels identical (acquisition fault)",
    "238.mat": "DE and FE channels identical (acquisition fault)",
    "191.mat": "clipped signals",
    "214.mat": "clipped signals",
    "215.mat": "clipped signals",
    "228.mat": "clipped signals",
    "229.mat": "clipped signals",
    "236.mat": "clipped signals",
    "237.mat": "clipped signals",
    # Packaging defects confirmed on the 2026-09-29 download (no array named
    # after the file number, and no RPM) — same defect as 3001.mat above.
    "174.mat": "contains only X173_* arrays — no X174 data (packaging bug)",
    "3002.mat": "variable named X057_DE_time; DE only, no FE, no RPM",
    "3003.mat": "variable named X058_DE_time; DE only, no FE, no RPM",
    "3004.mat": "variable named X059_DE_time; DE only, no FE, no RPM",
    "3005.mat": "variable named X048_DE_time; DE only, no FE, no RPM",
    "3006.mat": "variable named X049_DE_time; DE only, no FE, no RPM",
    "3007.mat": "variable named X050_DE_time; DE only, no FE, no RPM",
    "3008.mat": "variable named X051_DE_time; DE only, no FE, no RPM",
}

# Approx RPM by load (hp) for CWRU when RPM missing
LOAD_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


@dataclass
class CWRURecord:
    path: Path
    signal: np.ndarray
    meta: SampleMeta


def _load_mat(path: Path) -> dict:
    from scipy.io import loadmat

    return loadmat(str(path), squeeze_me=True)


def _find_time_key(mat: dict, prefer: str = "DE", file_number: str | None = None) -> str | None:
    """Array for this file's own number (99.mat also ships X098_* arrays)."""
    keys = [k for k in mat.keys() if not k.startswith("__") and k.endswith("_time")]
    if file_number is not None:
        want = f"X{int(file_number):03d}_{prefer}_time"
        return want if want in keys else None
    for k in keys:
        if prefer in k:
            return k
    return keys[0] if keys else None


def _find_rpm(mat: dict, file_number: str | None = None) -> float | None:
    keys = [k for k in mat.keys() if not k.startswith("__") and "RPM" in k.upper()]
    if file_number is not None:
        keys = [k for k in keys if k.upper() == f"X{int(file_number):03d}RPM"]
    for k in keys:
        try:
            return float(np.asarray(mat[k]).reshape(-1)[0])
        except Exception:
            pass
    return None


def parse_cwru_filename(name: str) -> dict:
    """Parse a CWRU file: numbered (via CATALOG) or fault-named (IR007_0, OR014@6_1, Normal_2).

    Returns fault_type, size_in, load_hp, or_position ('@3'/'@6'/'@12' or None),
    subset (None for fault-named files), stem.
    """
    import re

    stem = Path(name).stem
    subset = None
    if stem.isdigit():
        if stem not in CATALOG:
            raise ValueError(f"CWRU file number {stem} not in catalog")
        subset, fault_name = CATALOG[stem]
    else:
        fault_name = stem
    info = {"fault_type": "Normal", "size_in": None, "load_hp": None,
            "or_position": None, "subset": subset, "stem": stem}
    m = re.match(r"^(IR|OR|B)(\d{3})(@\d+)?_(\d)$", fault_name, re.IGNORECASE)
    if m:
        info["fault_type"] = m.group(1).upper()
        info["size_in"] = int(m.group(2)) / 1000.0
        info["or_position"] = m.group(3)
        info["load_hp"] = int(m.group(4))
        return info
    m = re.match(r"^normal_(\d)$", fault_name, re.IGNORECASE)
    if m:
        info["load_hp"] = int(m.group(1))
        return info
    raise ValueError(f"Cannot parse CWRU file name {name!r} (fault name {fault_name!r})")


def fault_identity_cwru(fault_type: str, size_in: float | None, or_position: str | None = None) -> str:
    """Physical fault identity — same EDM notch across loads (and 12k/48k) shares identity.

    Outer-race notches at @3/@6/@12 are distinct physical faults.
    """
    if fault_type in ("Normal", "normal"):
        return "cwru_healthy"
    parts = ["cwru", fault_type, f"{size_in:.3f}" if size_in else "unk"]
    if or_position:
        parts.append(or_position)
    return "_".join(parts)


def iter_cwru(
    root: str | Path,
    include_known_bad: bool = False,
    channel: str = "DE",
    subsets: tuple[str, ...] | None = ("normal", "12k_DE"),
    target_fs: float | None = None,
    assume_fs: float = 48000.0,
) -> Iterator[CWRURecord]:
    """Yield records under root.

    subsets: catalog subsets to keep (numbered files only; None keeps all).
    Default is the standard 12k drive-end benchmark plus normal baseline.
    target_fs: resample every record to this rate (normal baseline is 48 kHz,
    12k_DE is 12 kHz — mixing them un-resampled makes fs a class cue).
    assume_fs: rate for fault-named files not in the catalog.
    """
    root = Path(root)
    files = sorted(root.rglob("*.mat"), key=lambda p: (p.parent.name, p.name))
    for path in files:
        reason = KNOWN_BAD.get(path.name)
        if reason and not include_known_bad:
            continue
        try:
            info = parse_cwru_filename(path.name)
        except ValueError:
            continue
        if info["subset"] is not None and subsets is not None and info["subset"] not in subsets:
            continue
        try:
            mat = _load_mat(path)
        except Exception:
            continue
        number = info["stem"] if info["stem"].isdigit() else None
        key = _find_time_key(mat, prefer=channel, file_number=number)
        if key is None:
            continue
        sig = np.asarray(mat[key], dtype=np.float64).reshape(-1)
        fs = SUBSET_FS[info["subset"]] if info["subset"] else assume_fs
        resampled_from = None
        if target_fs is not None and abs(fs - target_fs) > 1e-6:
            from vibedge.deployment_view import resample_signal

            sig, _ = resample_signal(sig, fs, target_fs)
            resampled_from, fs = fs, float(target_fs)
        rpm = _find_rpm(mat, file_number=number)
        if rpm is None and info["load_hp"] is not None:
            rpm = float(LOAD_RPM.get(info["load_hp"], 1772))
        label = map_cwru_fault_type(info["fault_type"])
        fid = fault_identity_cwru(info["fault_type"], info["size_in"], info["or_position"])
        # the faulted bearing is on the fan end for 12k_FE; geometry follows the channel
        geo = CWRU_6205_DE if channel == "DE" else CWRU_6203_FE
        meta = SampleMeta(
            dataset="cwru",
            file_id=path.name,
            fault_id=fid,
            label=label,
            severity=info["size_in"],
            load_hp=info["load_hp"],
            rpm=rpm,
            fs=fs,
            channel=channel,
            geometry={"n": geo.n, "d": geo.d, "D": geo.D, "phi_deg": geo.phi_deg},
            extras={
                "subset": info["subset"],
                "or_position": info["or_position"],
                "fs_documented": info["subset"] not in (None, "normal"),
                "fs_note": "48kHz inferred from record length for baselines; not documented",
                "resampled_from_hz": resampled_from,
                "known_bad": reason,
            },
        )
        yield CWRURecord(path=path, signal=sig, meta=meta)


def load_cwru_directory(
    root: str | Path,
    include_known_bad: bool = False,
    channel: str = "DE",
    **kwargs,
) -> list[CWRURecord]:
    return list(iter_cwru(root, include_known_bad=include_known_bad, channel=channel, **kwargs))
