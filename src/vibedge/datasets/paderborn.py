"""Paderborn KAt Bearing Data Center loader.

Artificial vs real damage split is the interesting generalisation axis.
Download index: groups.uni-paderborn.de/kat/BearingDataCenter/
Licence: CC BY-NC 4.0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from vibedge.features.bearing import PADERBORN_6203
from vibedge.taxonomy import SampleMeta, map_paderborn

PADERBORN_FS = 64_000.0

# Bearing codes: K0xx healthy, KA outer ring, KI inner ring, KB combined (excluded).
HEALTHY_PREFIXES = ("K001", "K002", "K003", "K004", "K005", "K006")

# Damage origin per bearing (Lessmeier et al. 2016, Tables 4-5).
ARTIFICIAL_DAMAGE = {"KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",
                     "KI01", "KI03", "KI05", "KI07", "KI08"}
REAL_DAMAGE = {"KA04", "KA15", "KA16", "KA22", "KA30", "KB23", "KB24", "KB27",
               "KI04", "KI14", "KI16", "KI17", "KI18", "KI21"}


@dataclass
class PaderbornRecord:
    path: Path
    signal: np.ndarray
    meta: SampleMeta


def _bearing_code(path: Path) -> str:
    # e.g. N09_M07_F10_KA04_1.mat → KA04
    stem = path.stem
    parts = stem.split("_")
    for p in parts:
        if p.startswith(("K0", "KA", "KI", "KB")):
            return p
    return stem[:4]


def _is_healthy(code: str) -> bool:
    return any(code.startswith(h[:4]) for h in HEALTHY_PREFIXES) or code.startswith("K0")


def _speed_rpm(path: Path) -> float | None:
    # N09_M07_F10_... → 900 rpm; N15 → 1500 rpm
    import re

    m = re.match(r"N(\d{2})_", path.stem)
    return float(m.group(1)) * 100.0 if m else None


def _vibration_signal(mat: dict, stem: str, channel: str) -> np.ndarray | None:
    # Each file holds one struct named after the file; its .Y lists channels.
    root = mat.get(stem)
    if root is None:
        cands = [v for k, v in mat.items() if not k.startswith("__")]
        root = cands[0] if len(cands) == 1 else None
    Y = getattr(root, "Y", None) if root is not None else mat.get("Y")
    if Y is None:
        return None
    for entry in np.atleast_1d(Y):
        if str(getattr(entry, "Name", "")) == channel:
            return np.asarray(entry.Data, dtype=np.float64).reshape(-1)
    return None


def load_paderborn_file(path: str | Path, channel: str = "vibration_1") -> PaderbornRecord | None:
    """Load one Paderborn .mat; None for KB* (combined damage) or unreadable files."""
    path = Path(path)
    code = _bearing_code(path)
    healthy = _is_healthy(code)
    try:
        label = map_paderborn(code, healthy=healthy)
    except ValueError:
        return None  # KB* combined damage — not a single-fault class
    try:
        from scipy.io import loadmat

        mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except Exception:
        return None
    sig = _vibration_signal(mat, path.stem, channel)
    if sig is None:
        return None
    geo = PADERBORN_6203
    meta = SampleMeta(
        dataset="paderborn",
        file_id=path.name,
        fault_id=f"paderborn_{code}",  # one physical bearing per code
        label=label,
        rpm=_speed_rpm(path),
        fs=PADERBORN_FS,
        geometry={"n": geo.n, "d": geo.d, "D": geo.D, "phi_deg": geo.phi_deg},
        extras={
            "bearing_code": code,
            "artificial": code in ARTIFICIAL_DAMAGE,
            "real_damage": code in REAL_DAMAGE,
            "operating_condition": "_".join(path.stem.split("_")[:3]),
            "licence": "CC BY-NC 4.0",
        },
    )
    return PaderbornRecord(path=path, signal=sig, meta=meta)


def iter_paderborn(
    root: str | Path,
    channel: str = "vibration_1",
) -> Iterator[PaderbornRecord]:
    for path in sorted(Path(root).rglob("*.mat")):
        rec = load_paderborn_file(path, channel)
        if rec is not None:
            yield rec


def load_paderborn(root: str | Path, **kwargs) -> list[PaderbornRecord]:
    return list(iter_paderborn(root, **kwargs))
