"""Deployment-sensor simulation: 6 kHz LPF, resample to 26.7 kSPS, MEMS noise.

Every result should be reportable on both pristine and deployment views.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd


def butter_lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 8) -> np.ndarray:
    if cutoff >= fs / 2:
        return x.astype(np.float64, copy=False)
    wn = cutoff / (fs / 2.0)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x.astype(np.float64))


def add_mems_noise(
    x: np.ndarray,
    fs: float,
    density_ug_per_sqrt_hz: float = 75.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Inject white noise matching IIS3DWB noise density (µg/√Hz → g)."""
    rng = rng or np.random.default_rng(0)
    # noise density in g/√Hz
    dens_g = density_ug_per_sqrt_hz * 1e-6
    # for white noise, std = dens * sqrt(fs/2) roughly (Nyquist bandwidth)
    sigma = dens_g * np.sqrt(fs / 2.0)
    return x.astype(np.float64) + rng.normal(0.0, sigma, size=x.shape)


def resample_signal(x: np.ndarray, fs_in: float, fs_out: float) -> tuple[np.ndarray, float]:
    """Resample using rational approximation via resample_poly."""
    if abs(fs_in - fs_out) < 1e-6:
        return x.astype(np.float64), fs_out
    # find small integer up/down
    ratio = fs_out / fs_in
    # limit denominator
    up = int(round(ratio * 1000))
    down = 1000
    g = gcd(up, down)
    up //= g
    down //= g
    # clamp extreme ratios
    if up > 500 or down > 500:
        # fallback: compute via length
        n_out = int(round(len(x) * fs_out / fs_in))
        t_in = np.arange(len(x)) / fs_in
        t_out = np.arange(n_out) / fs_out
        y = np.interp(t_out, t_in, x.astype(np.float64))
        return y, fs_out
    y = resample_poly(x.astype(np.float64), up, down)
    return y, fs_out


def to_deployment_view(
    x: np.ndarray,
    fs: float,
    lpf_hz: float = 6000.0,
    target_fs: float = 26700.0,
    noise_density_ug_per_sqrt_hz: float = 75.0,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Pristine lab signal → simulated IIS3DWB deployment capture."""
    y = butter_lowpass(x, fs, min(lpf_hz, 0.45 * fs))
    y, fs_out = resample_signal(y, fs, target_fs)
    if add_noise:
        y = add_mems_noise(y, fs_out, noise_density_ug_per_sqrt_hz, rng=rng)
    return y, fs_out
