"""Unified 6-class label space and per-dataset mappings.

Classes:
  healthy, imbalance, misalignment, bearing_IR, bearing_OR, bearing_ball

Severity / load are metadata, not classes.
Imbalance evaluation is scoped to held-out severity on MaFaulDa only —
fault-identity and cross-dataset splits are structurally impossible for
imbalance on public data (single-rig).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class FaultClass(str, Enum):
    HEALTHY = "healthy"
    IMBALANCE = "imbalance"
    MISALIGNMENT = "misalignment"
    BEARING_IR = "bearing_IR"
    BEARING_OR = "bearing_OR"
    BEARING_BALL = "bearing_ball"


CLASSES: list[str] = [c.value for c in FaultClass]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}

# Bearing subset used for cross-dataset transfer
BEARING_CLASSES: list[str] = [
    FaultClass.HEALTHY.value,
    FaultClass.BEARING_IR.value,
    FaultClass.BEARING_OR.value,
    FaultClass.BEARING_BALL.value,
]


@dataclass
class SampleMeta:
    """Metadata attached to every window / recording."""

    dataset: str
    file_id: str
    fault_id: str  # physical fault / bearing identity for leakage-safe splits
    label: str  # unified class name
    severity: float | None = None
    load_hp: float | None = None
    rpm: float | None = None
    fs: float = 0.0
    channel: str = "DE"
    geometry: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "file_id": self.file_id,
            "fault_id": self.fault_id,
            "label": self.label,
            "severity": self.severity,
            "load_hp": self.load_hp,
            "rpm": self.rpm,
            "fs": self.fs,
            "channel": self.channel,
            "geometry": self.geometry,
            **self.extras,
        }


# ---------------------------------------------------------------------------
# Per-dataset raw → unified mappings
# ---------------------------------------------------------------------------

# CWRU: fault type from filename / metadata conventions
CWRU_MAP: dict[str, str] = {
    "Normal": FaultClass.HEALTHY.value,
    "normal": FaultClass.HEALTHY.value,
    "IR": FaultClass.BEARING_IR.value,
    "InnerRace": FaultClass.BEARING_IR.value,
    "OR": FaultClass.BEARING_OR.value,
    "OuterRace": FaultClass.BEARING_OR.value,
    "B": FaultClass.BEARING_BALL.value,
    "Ball": FaultClass.BEARING_BALL.value,
}

# MFPT
MFPT_MAP: dict[str, str] = {
    "Normal": FaultClass.HEALTHY.value,
    "InnerRace": FaultClass.BEARING_IR.value,
    "OuterRace": FaultClass.BEARING_OR.value,
}

# MaFaulDa folder / class name fragments → unified
# Official taxonomy has 42 classes; we collapse.
MAFAULDA_MAP_RULES: list[tuple[str, str]] = [
    ("normal", FaultClass.HEALTHY.value),
    ("imbalance", FaultClass.IMBALANCE.value),
    ("horizontal-misalignment", FaultClass.MISALIGNMENT.value),
    ("vertical-misalignment", FaultClass.MISALIGNMENT.value),
    ("underhang/ball_fault", FaultClass.BEARING_BALL.value),
    ("overhang/ball_fault", FaultClass.BEARING_BALL.value),
    ("underhang/cage_fault", FaultClass.BEARING_OR.value),  # cage ≈ outer-ish; documented
    ("overhang/cage_fault", FaultClass.BEARING_OR.value),
    ("underhang/outer_race", FaultClass.BEARING_OR.value),
    ("overhang/outer_race", FaultClass.BEARING_OR.value),
]


def map_mafaulda_path(relpath: str) -> str:
    """Map a MaFaulDa relative path / class folder to unified label."""
    p = relpath.replace("\\", "/").lower()
    for needle, label in MAFAULDA_MAP_RULES:
        if needle in p:
            return label
    raise ValueError(f"Cannot map MaFaulDa path to unified label: {relpath!r}")


def map_cwru_fault_type(fault_type: str) -> str:
    if fault_type not in CWRU_MAP:
        raise ValueError(f"Unknown CWRU fault type: {fault_type!r}")
    return CWRU_MAP[fault_type]


def map_mfpt_fault_type(fault_type: str) -> str:
    if fault_type not in MFPT_MAP:
        raise ValueError(f"Unknown MFPT fault type: {fault_type!r}")
    return MFPT_MAP[fault_type]


def map_paderborn(damage_mode: str, healthy: bool = False) -> str:
    """Map Paderborn damage codes to unified labels.

    Paderborn codes: K0xx healthy, KA = outer ring (Außenring), KI = inner
    ring (Innenring), KB = COMBINED inner+outer damage. Paderborn has no
    rolling-element faults, so KB has no single-fault class and raises.
    """
    if healthy:
        return FaultClass.HEALTHY.value
    code = damage_mode.upper()
    if code.startswith("KB"):
        raise ValueError(
            f"Paderborn {damage_mode!r} is combined IR+OR damage; no single class in the 6-class taxonomy"
        )
    if "KI" in code or code.startswith("IR"):
        return FaultClass.BEARING_IR.value
    if "KA" in code or code.startswith("OR"):
        return FaultClass.BEARING_OR.value
    raise ValueError(f"Cannot map Paderborn damage mode: {damage_mode!r}")


def labels_to_indices(labels: list[str] | np.ndarray) -> np.ndarray:
    out = np.empty(len(labels), dtype=np.int64)
    for i, lab in enumerate(labels):
        if lab not in CLASS_TO_IDX:
            raise KeyError(f"Unknown unified label: {lab!r}")
        out[i] = CLASS_TO_IDX[lab]
    return out


def indices_to_labels(indices: np.ndarray) -> list[str]:
    return [IDX_TO_CLASS[int(i)] for i in indices]


# Protocol feasibility matrix (dataset × protocol)
# True = protocol is meaningful for that dataset's claim.
PROTOCOL_FEASIBILITY: dict[str, dict[int, str]] = {
    # value: "ok" | "weak" | "impossible" | "bearing_only"
    "synthetic": {0: "ok", 1: "ok", 2: "ok", 3: "ok", 4: "ok", 5: "ok"},
    "mafaulda": {
        0: "ok",
        1: "ok",
        2: "weak",
        3: "impossible",  # one physical object per class
        4: "ok",  # held-out severity — the honest imbalance protocol
        5: "bearing_only",  # imbalance cannot transfer
    },
    "cwru": {
        0: "ok",
        1: "weak",
        2: "weak",  # same physical bearing across loads
        3: "ok",
        4: "ok",
        5: "bearing_only",
    },
    # one conservative identity per fault type (see datasets/mfpt.py)
    "mfpt": {0: "ok", 1: "ok", 2: "weak", 3: "impossible", 4: "weak", 5: "bearing_only"},
    "paderborn": {
        0: "ok",
        1: "ok",
        2: "ok",
        3: "ok",
        4: "ok",
        5: "bearing_only",
    },
}
