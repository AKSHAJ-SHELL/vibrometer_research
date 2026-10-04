"""Time-domain scalar features (10). Prefer dimensionless ones for transfer.

Kurtosis convention: PEARSON (Gaussian = 3, unit sine = 1.5), population
moments (``bias=True``), as quoted in the bearing-diagnostics literature.
scipy's default is Fisher (excess, Gaussian = 0) — do not mix the two.
Skewness likewise uses population moments.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


KURTOSIS_CONVENTION = "pearson"

FEATURE_NAMES = [
    "rms",
    "peak",
    "peak_to_peak",
    "std",
    "kurtosis",
    "skewness",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
]


def time_domain_features(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {n: 0.0 for n in FEATURE_NAMES}
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(abs_x))
    p2p = float(np.ptp(x))
    std = float(np.std(x))
    mean_abs = float(np.mean(abs_x))
    mean_sqrt_abs = float(np.mean(np.sqrt(abs_x))) ** 2  # square of mean sqrt
    eps = 1e-12
    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": p2p,
        "std": std,
        "kurtosis": float(kurtosis(x, fisher=False, bias=True)),
        "skewness": float(skew(x, bias=True)),
        "crest_factor": peak / (rms + eps),
        "shape_factor": rms / (mean_abs + eps),
        "impulse_factor": peak / (mean_abs + eps),
        "clearance_factor": peak / (mean_sqrt_abs + eps),
    }
