"""Feature extraction pipeline — two-timescale, speed-aware."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vibedge.features.bearing import GENERIC_6205, BearingGeometry
from vibedge.features.envelope import (
    ENVELOPE_FEATURE_NAMES,
    compute_envelope_spectrum,
    envelope_fault_features,
    envelope_resolution_hz,
)
from vibedge.features.spectral import (
    BROADBAND_FEATURE_NAMES,
    ORDER_FEATURE_NAMES,
    broadband_features,
    order_amplitudes,
    rfft_mag,
    velocity_rms_iso,
)
from vibedge.features.time_domain import FEATURE_NAMES as TIME_NAMES
from vibedge.features.time_domain import time_domain_features
from vibedge.speed import SPEED_FREE_FEATURES, SpeedEstimate, estimate_speed

# Canonical ordered feature vector (~38 dims + flags)
FEATURE_ORDER: list[str] = (
    TIME_NAMES
    + ["iso_velocity_rms"]
    + ORDER_FEATURE_NAMES
    + ENVELOPE_FEATURE_NAMES
    + BROADBAND_FEATURE_NAMES  # 5 of the broadband set used in table; keep all 7
)

# Trim broadband to 5 as in design table if desired — keep all, document extras.
SPEED_DEPENDENT: set[str] = set(ORDER_FEATURE_NAMES) | set(ENVELOPE_FEATURE_NAMES)

ALL_GROUPS: list[str] = [
    "time_domain",
    "broadband",
    "iso_severity",
    "order_domain",
    "envelope_bands",
    "sidebands",
]

# Named feature sets compared in C1. `full` = 10 + 7 + 1 + 6 + 12 + 4 = 40 dims.
FEATURE_SET_GROUPS: dict[str, list[str]] = {
    "time_only": ["time_domain", "broadband"],
    "envelope_ratio": [
        "time_domain",
        "broadband",
        "envelope_bands",
        "sidebands",
        "order_domain",
    ],
    "full": [
        "time_domain",
        "broadband",
        "iso_severity",
        "order_domain",
        "envelope_bands",
        "sidebands",
    ],
}


@dataclass
class FeatureResult:
    values: dict[str, float]
    vector: np.ndarray
    names: list[str]
    speed: SpeedEstimate
    used_speed_features: bool
    extras: dict[str, Any] = field(default_factory=dict)

    def as_speed_free(self) -> np.ndarray:
        idx = [i for i, n in enumerate(self.names) if n in SPEED_FREE_FEATURES]
        return self.vector[idx]

    def subset(self, groups: list[str]) -> tuple[np.ndarray, list[str]]:
        """Select feature groups by name."""
        wanted: list[str] = []
        if "time_domain" in groups:
            wanted.extend(TIME_NAMES)
        if "iso_severity" in groups:
            wanted.append("iso_velocity_rms")
        if "order_domain" in groups:
            wanted.extend(ORDER_FEATURE_NAMES)
        if "envelope_bands" in groups or "sidebands" in groups:
            # envelope_bands = harmonic energies; sidebands = sideband feats
            if "envelope_bands" in groups:
                wanted.extend([n for n in ENVELOPE_FEATURE_NAMES if "sideband" not in n])
            if "sidebands" in groups:
                wanted.extend([n for n in ENVELOPE_FEATURE_NAMES if "sideband" in n])
        if "broadband" in groups:
            wanted.extend(BROADBAND_FEATURE_NAMES)
        # unique preserve order
        seen = set()
        names = []
        for n in wanted:
            if n not in seen and n in self.values:
                seen.add(n)
                names.append(n)
        vec = np.array([self.values[n] for n in names], dtype=np.float64)
        return vec, names


def extract_features(
    x: np.ndarray,
    fs: float,
    geo: BearingGeometry | None = None,
    fr_hz: float | None = None,
    speed_confidence_threshold: float = 0.35,
    resonance_band: tuple[float, float] = (2000.0, 5000.0),
    slow_seconds: float = 4.0,
    envelope_decim: int = 16,
    fault_band_tol: float = 0.02,
    search_hz: tuple[float, float] = (10.0, 60.0),
    groups: list[str] | None = None,
) -> FeatureResult:
    """Feature extraction on one window / recording segment.

    ``groups`` limits computation to those feature groups (default: all), so
    per-feature-set extraction cost can be timed honestly.
    """
    geo = geo or GENERIC_6205
    x = np.asarray(x, dtype=np.float64)
    groups = list(ALL_GROUPS) if groups is None else list(groups)
    unknown = set(groups) - set(ALL_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups {sorted(unknown)}")
    want_env = "envelope_bands" in groups or "sidebands" in groups
    want_order = "order_domain" in groups
    need_speed = want_env or want_order

    # --- speed ---
    if fr_hz is not None and fr_hz > 0:
        speed = SpeedEstimate(fr_hz=float(fr_hz), confidence=1.0, method="provided")
    elif need_speed:
        speed = estimate_speed(x, fs, search_hz=search_hz)
    else:
        speed = SpeedEstimate(fr_hz=0.0, confidence=0.0, method="not_needed")

    use_speed = speed.confidence >= speed_confidence_threshold and speed.fr_hz > 0

    values: dict[str, float] = {}
    extras: dict[str, Any] = {
        "envelope_resolution_hz": envelope_resolution_hz(fs, slow_seconds, envelope_decim),
        "n_samples": len(x),
    }

    # --- fast path: time + broadband + ISO ---
    if "time_domain" in groups:
        values.update(time_domain_features(x))
    if "broadband" in groups or want_order:
        freqs, mag = rfft_mag(x, fs)
        if "broadband" in groups:
            values.update(broadband_features(freqs, mag, fs))
    if "iso_severity" in groups:
        values["iso_velocity_rms"] = velocity_rms_iso(x, fs)

    # --- order domain ---
    if want_order:
        if use_speed:
            values.update(order_amplitudes(freqs, mag, speed.fr_hz, tol=fault_band_tol))
        else:
            for n in ORDER_FEATURE_NAMES:
                values[n] = 0.0

    # --- slow path: envelope ---
    if want_env:
        env_spec = compute_envelope_spectrum(
            x,
            fs,
            band=resonance_band,
            decim=envelope_decim,
            target_seconds=slow_seconds if len(x) / fs < slow_seconds else None,
        )
        extras["env_fs"] = env_spec.fs_env
        if use_speed:
            values.update(
                envelope_fault_features(
                    env_spec, speed.fr_hz, geo, tol=fault_band_tol, normalize=True
                )
            )
        else:
            for n in ENVELOPE_FEATURE_NAMES:
                values[n] = 0.0

    # assemble vector in canonical order (only keys present)
    names = [n for n in FEATURE_ORDER if n in values]
    # add any extras
    for n in values:
        if n not in names:
            names.append(n)
    vector = np.array([values[n] for n in names], dtype=np.float64)

    return FeatureResult(
        values=values,
        vector=vector,
        names=names,
        speed=speed,
        used_speed_features=use_speed and need_speed,
        extras=extras,
    )


from vibedge.features import bearing as bearing  # noqa: E402 — re-export
from vibedge.features import envelope as envelope
from vibedge.features import spectral as spectral
from vibedge.features import time_domain as time_domain

__all__ = [
    "extract_features",
    "FeatureResult",
    "FEATURE_ORDER",
    "FEATURE_SET_GROUPS",
    "ALL_GROUPS",
    "SPEED_DEPENDENT",
    "bearing",
    "envelope",
    "spectral",
    "time_domain",
]
