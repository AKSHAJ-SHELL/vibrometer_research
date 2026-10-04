"""Cost column for the EMC² table: what each feature set costs on the device.

Per feature set: feature count, feature-vector bytes (float32 on the MCU),
model parameter count, and host-side extraction time per window. Host
timing is a relative ranking between feature sets, not an MCU latency —
MCU cycles come from the phase-2 on-device measurements.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vibedge.features import FEATURE_SET_GROUPS, extract_features
from vibedge.features.bearing import BearingGeometry
from vibedge.models import fit_model, model_param_count

FLOAT32_BYTES = 4


def feature_vector_bytes(n_features: int, dtype_bytes: int = FLOAT32_BYTES) -> int:
    return int(n_features) * int(dtype_bytes)


def feature_count(groups: list[str], x: np.ndarray, fs: float, fr_hz: float,
                  geo: BearingGeometry | None = None) -> int:
    _, names = extract_features(x, fs, geo=geo, fr_hz=fr_hz, groups=groups).subset(groups)
    return len(names)


def extraction_time_per_window(
    x: np.ndarray,
    fs: float,
    groups: list[str],
    fr_hz: float | None = None,
    geo: BearingGeometry | None = None,
    repeats: int = 5,
) -> float:
    """Median wall-clock seconds for one extract_features call on window x."""
    extract_features(x, fs, geo=geo, fr_hz=fr_hz, groups=groups)  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        extract_features(x, fs, geo=geo, fr_hz=fr_hz, groups=groups)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def cost_table(
    x_window: np.ndarray,
    fs: float,
    fr_hz: float | None,
    X_by_set: dict[str, np.ndarray] | None = None,
    y: np.ndarray | None = None,
    model_names: tuple[str, ...] = ("logreg", "gbdt"),
    geo: BearingGeometry | None = None,
    feature_sets: dict[str, list[str]] | None = None,
    repeats: int = 5,
) -> list[dict[str, Any]]:
    """One row per feature set. Model params need (X_by_set, y) to fit on."""
    feature_sets = feature_sets or FEATURE_SET_GROUPS
    rows = []
    for name, groups in feature_sets.items():
        n_feat = feature_count(groups, x_window, fs, fr_hz or 0.0, geo)
        row: dict[str, Any] = {
            "feature_set": name,
            "groups": list(groups),
            "n_features": n_feat,
            "feature_bytes_float32": feature_vector_bytes(n_feat),
            "extract_s_per_window": extraction_time_per_window(
                x_window, fs, groups, fr_hz=fr_hz, geo=geo, repeats=repeats
            ),
            "window_s": len(x_window) / fs,
            "timing_note": "host CPU, median of repeats; relative only",
        }
        if X_by_set is not None and y is not None and name in X_by_set:
            for m in model_names:
                row[f"params_{m}"] = model_param_count(fit_model(m, X_by_set[name], y))
        rows.append(row)
    return rows
