"""MaFaulDa loader — primary dataset (only public source with imbalance).

Official host www02.smt.ufrj.br was unreachable during research.
Use Kaggle mirrors; licence not stated — say so, do not invent one.

Imbalance evaluation: held-out severity only (protocol 4). Fault-identity
and cross-dataset splits are structurally impossible for imbalance here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from vibedge.features.bearing import MAFAULDA_BEARING
from vibedge.taxonomy import SampleMeta, map_mafaulda_path

# Nominal sampling rate
MAFAULDA_FS = 50_000.0

# Channel layout commonly used in mirrors (8 channels):
# 0 tachometer, 1–3 underhang accel XYZ, 4–6 overhang accel XYZ, 7 microphone
TACHO_CH = 0
DEFAULT_ACCEL_CH = 1  # underhang axial/radial — mirror-dependent; documented in meta


@dataclass
class MaFaulDaRecord:
    path: Path
    signal: np.ndarray
    tacho: np.ndarray | None
    meta: SampleMeta


def _parse_severity_grams(path: Path) -> float | None:
    """Extract imbalance mass in grams from path like imbalance/10g/."""
    s = str(path).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*g", s)
    if m:
        return float(m.group(1))
    m = re.search(r"imbalance[/\\](\d+)", s)
    if m:
        return float(m.group(1))
    return None


def _fault_id_mafaulda(rel: str, label: str, severity: float | None) -> str:
    """Physical configuration on this single rig.

    Bearing faults: one faulty bearing per (position, fault) folder — the
    0g/6g/20g/35g subfolders are imbalance mass ADDED to the same bearing, not
    a different fault. Imbalance: one config per mass. Misalignment: one per
    (direction, offset). Healthy: one.
    """
    parts = rel.replace("\\", "/").split("/")
    if label == "healthy":
        return "mafaulda_healthy"
    if parts[0] in ("underhang", "overhang"):
        return f"mafaulda_{parts[0]}_{parts[1]}"
    if label == "imbalance":
        return f"mafaulda_imbalance_{severity:g}g"
    return f"mafaulda_{parts[0]}_{parts[1]}"


def load_mafaulda_file(
    root: str | Path, path: str | Path, accel_channel: int = DEFAULT_ACCEL_CH
) -> MaFaulDaRecord | None:
    """Load one MaFaulDa CSV; None if it can't be mapped or read."""
    root, path = Path(root), Path(path)
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    try:
        label = map_mafaulda_path(rel)
    except ValueError:
        return None
    try:
        import pandas as pd

        data = pd.read_csv(path, header=None).values
    except Exception:
        return None
    if data.ndim == 1:
        sig, tacho = data.astype(np.float64), None
    else:
        if data.shape[1] <= accel_channel:
            return None
        sig = data[:, accel_channel].astype(np.float64)
        tacho = data[:, TACHO_CH].astype(np.float64) if data.shape[1] > TACHO_CH else None

    severity = _parse_severity_grams(path) if label == "imbalance" else None
    if label == "misalignment":
        m = re.search(r"(\d+(?:\.\d+)?)\s*mm", str(path).lower())
        severity = float(m.group(1)) if m else severity

    fid = _fault_id_mafaulda(rel, label, severity)
    # file stem is the measured rotation frequency in Hz (e.g. 12.288.csv)
    try:
        fr_name = float(path.name.split(".csv")[0])
    except ValueError:
        fr_name = None
    geo = MAFAULDA_BEARING
    meta = SampleMeta(
        dataset="mafaulda",
        file_id=rel,
        fault_id=fid,
        label=label,
        severity=severity,
        rpm=fr_name * 60.0 if fr_name else None,
        fs=MAFAULDA_FS,
        channel=f"ch{accel_channel}",
        geometry={"n": geo.n, "d": geo.d, "D": geo.D, "phi_deg": geo.phi_deg},
        extras={
            "licence": "not stated",
            "source": "UFRJ official host",
            "has_tacho": tacho is not None,
            "fr_from_filename_hz": fr_name,
            "subfault": "cage" if "cage_fault" in rel else None,
        },
    )
    return MaFaulDaRecord(path=path, signal=sig, tacho=tacho, meta=meta)


def mafaulda_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(list(root.rglob("*.csv")) + list(root.rglob("*.csv.gz")))


def iter_mafaulda(
    root: str | Path,
    accel_channel: int = DEFAULT_ACCEL_CH,
    max_files: int | None = None,
) -> Iterator[MaFaulDaRecord]:
    count = 0
    for path in mafaulda_files(root):
        if max_files is not None and count >= max_files:
            break
        rec = load_mafaulda_file(root, path, accel_channel)
        if rec is None:
            continue
        count += 1
        yield rec


def load_mafaulda(root: str | Path, **kwargs) -> list[MaFaulDaRecord]:
    return list(iter_mafaulda(root, **kwargs))


# Honest imbalance protocol defaults (grams)
IMBALANCE_TRAIN_SEVERITIES = [6.0, 10.0, 20.0, 30.0, 35.0]
IMBALANCE_TEST_SEVERITIES = [15.0, 25.0]
IMBALANCE_EXTRAPOLATE_TEST = [35.0]  # train without max, test max
