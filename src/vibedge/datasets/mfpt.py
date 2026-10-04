"""MFPT loader. Canonical mfpt.org URL is dead; use third-party mirrors.

Only the labelled rig folders are used (1 baseline, 2-3 outer race, 4 inner
race). "5 - Analyses" duplicates a folder-4 file; "6 - Real World Examples"
are other machines with no class in this taxonomy.

Fault identity is per fault TYPE: MFPT does not document whether the
different-load recordings use distinct bearings, so they are grouped
conservatively. Protocol 3 is therefore infeasible on MFPT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from vibedge.features.bearing import MFPT_BEARING
from vibedge.taxonomy import SampleMeta, map_mfpt_fault_type


@dataclass
class MFPTRecord:
    path: Path
    signal: np.ndarray
    meta: SampleMeta


def _infer_label(path: Path) -> str | None:
    """Label from the rig folder name; None for folders outside the taxonomy."""
    folder = path.parent.name.lower()
    if "analyses" in folder or "real world" in folder:
        return None
    if "baseline" in folder:
        return "Normal"
    if "inner race" in folder:
        return "InnerRace"
    if "outer race" in folder:
        return "OuterRace"
    return None


def iter_mfpt(root: str | Path) -> Iterator[MFPTRecord]:
    root = Path(root)
    files = sorted(list(root.rglob("*.mat")) + list(root.rglob("*.MAT")))
    for path in files:
        ftype = _infer_label(path)
        if ftype is None:
            continue
        try:
            from scipy.io import loadmat

            mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        except Exception:
            continue
        b = mat.get("bearing")
        if b is None:
            continue
        sig = np.asarray(b.gs, dtype=np.float64).reshape(-1)
        fs = float(b.sr)
        fr = float(b.rate)  # shaft rate, Hz
        load = float(b.load)
        label = map_mfpt_fault_type(ftype)
        geo = MFPT_BEARING
        meta = SampleMeta(
            dataset="mfpt",
            file_id=path.name,
            fault_id=f"mfpt_{ftype}",
            label=label,
            load_hp=None,
            rpm=fr * 60.0,
            fs=fs,
            geometry={"n": geo.n, "d": geo.d, "D": geo.D, "phi_deg": geo.phi_deg},
            extras={"canonical_url": "dead — third-party mirror", "load_lbs": load,
                    "fr_hz": fr, "folder": path.parent.name},
        )
        yield MFPTRecord(path=path, signal=sig, meta=meta)


def load_mfpt(root: str | Path) -> list[MFPTRecord]:
    return list(iter_mfpt(root))
