"""Energy integration for power traces (INA219 measurements, phase 2).

E = ∫ P dt by the trapezoid rule. Trapezoid smears a step over one sample
interval — the error at a jump of ΔP is ≈ ΔP·dt/2 — so sample well below the
duration of the shortest power state you need to resolve.
"""

from __future__ import annotations

import numpy as np


def integrate_energy_j(t_s: np.ndarray, p_w: np.ndarray) -> float:
    """Trapezoid energy (J) of power samples p_w (W) at times t_s (s)."""
    t = np.asarray(t_s, dtype=np.float64)
    p = np.asarray(p_w, dtype=np.float64)
    if t.shape != p.shape or t.ndim != 1:
        raise ValueError(f"t and p must be equal-length 1-D arrays, got {t.shape} and {p.shape}")
    if len(t) < 2:
        raise ValueError("need at least two samples")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(p))):
        raise ValueError("non-finite sample in power trace")
    if np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return float(np.trapezoid(p, t))


def power_from_iv(current_a: np.ndarray, bus_voltage_v: np.ndarray) -> np.ndarray:
    """P = I·V per sample (INA219 reports bus voltage and shunt current)."""
    return np.asarray(current_a, dtype=np.float64) * np.asarray(bus_voltage_v, dtype=np.float64)


def max_step_error_j(delta_p_w: float, dt_s: float) -> float:
    """Worst-case trapezoid error from one unresolved power step."""
    return abs(delta_p_w) * dt_s / 2.0
