"""Spectral, order-domain, broadband, and ISO severity features."""

from __future__ import annotations

import numpy as np
from scipy.signal import windows


def rfft_mag(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    w = windows.hann(n, sym=False)
    X = np.fft.rfft(x.astype(np.float64) * w)
    mag = np.abs(X) * 2.0 / (np.sum(w) + 1e-12)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, mag


def single_sided_amplitude(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular-window single-sided amplitude spectrum: 2·|X[k]|/N.

    DC (and Nyquist, for even N) are not doubled. A unit sine exactly on a
    bin reads 1.0; off-bin it reads less (spectral leakage) — e.g. 50 Hz at
    fs=1000, N=4096 peaks at 0.935 in bin 205.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    amp = 2.0 * np.abs(np.fft.rfft(x)) / n
    amp[0] /= 2.0
    if n % 2 == 0:
        amp[-1] /= 2.0
    return np.fft.rfftfreq(n, d=1.0 / fs), amp


def spectral_energy(x: np.ndarray) -> float:
    """(1/N)·Σ|FFT(x)|² from the one-sided rFFT — equals Σx² by Parseval."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    p = np.abs(np.fft.rfft(x)) ** 2
    # interior bins stand for a ± pair; DC (and Nyquist for even n) are single
    w = np.full(p.shape, 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    return float(np.sum(w * p) / n)


def velocity_rms_iso(x_acc: np.ndarray, fs: float, f_lo: float = 10.0, f_hi: float = 1000.0) -> float:
    """ISO-structure velocity RMS over 10–1000 Hz from acceleration.

    Integrate in frequency domain: V(f) = A(f) / (j 2π f), then RMS of band.
    Caveat: MEMS 1/ω noise amplification at low end — reported honestly.
    """
    freqs, mag = rfft_mag(x_acc, fs)
    # exclude DC
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return 0.0
    # approximate velocity amplitude spectrum
    v_mag = np.zeros_like(mag)
    nz = freqs > 0
    v_mag[nz] = mag[nz] / (2.0 * np.pi * freqs[nz])
    # Parseval-ish band RMS from one-sided spectrum
    band = v_mag[mask]
    return float(np.sqrt(np.mean(band**2)))


def order_amplitudes(
    freqs: np.ndarray,
    mag: np.ndarray,
    fr: float,
    orders: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
    tol: float = 0.02,
) -> dict[str, float]:
    """Amplitudes at k×f_r normalised by total spectral energy."""
    total = float(np.sum(mag**2)) + 1e-12
    out: dict[str, float] = {}
    for o in orders:
        center = o * fr
        lo, hi = center * (1 - tol), center * (1 + tol)
        m = (freqs >= lo) & (freqs <= hi)
        e = float(np.sum(mag[m] ** 2)) if np.any(m) else 0.0
        key = f"order_{o:g}x"
        out[key] = e / total
    return out


def broadband_features(
    freqs: np.ndarray,
    mag: np.ndarray,
    fs: float,
) -> dict[str, float]:
    """Spectral centroid, entropy, flatness, rolloff, band-energy ratios."""
    p = mag**2
    p_sum = float(np.sum(p)) + 1e-12
    pn = p / p_sum
    centroid = float(np.sum(freqs * pn))
    # entropy
    pn_safe = pn[pn > 0]
    entropy = float(-np.sum(pn_safe * np.log2(pn_safe + 1e-300)))
    # flatness: geometric / arithmetic mean
    geo = float(np.exp(np.mean(np.log(mag + 1e-12))))
    arith = float(np.mean(mag) + 1e-12)
    flatness = geo / arith
    # rolloff: freq below which 85% energy
    csum = np.cumsum(p)
    rolloff_idx = int(np.searchsorted(csum, 0.85 * csum[-1]))
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    def band_e(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return float(np.sum(p[m])) / p_sum

    return {
        "spectral_centroid": centroid,
        "spectral_entropy": entropy,
        "spectral_flatness": flatness,
        "spectral_rolloff": rolloff,
        "band_energy_1_2k": band_e(1000, 2000),
        "band_energy_2_4k": band_e(2000, 4000),
        "band_energy_4_6k": band_e(4000, min(6000, fs / 2 - 1)),
    }


ORDER_FEATURE_NAMES = [f"order_{o:g}x" for o in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)]
BROADBAND_FEATURE_NAMES = [
    "spectral_centroid",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_rolloff",
    "band_energy_1_2k",
    "band_energy_2_4k",
    "band_energy_4_6k",
]
