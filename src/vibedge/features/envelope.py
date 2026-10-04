"""Two-timescale envelope analysis.

Fast path (0.15 s @ 26.7 kSPS): broadband / impulsiveness — always-on gate.
Slow path: bandpass → Hilbert → envelope → decimate → accumulate ~4 s →
envelope FFT with ~0.25 Hz resolution so ±2% fault bands and sidebands resolve.

Band selection: fixed resonance band by default; optional kurtogram for offline
band pick (too expensive for always-on MCU path).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, decimate, windows

from vibedge.features.bearing import (
    BearingGeometry,
    band_energy,
    bearing_freqs,
    harmonic_band_energies,
    sideband_pair_energy,
)


@dataclass
class EnvelopeSpectrum:
    freqs: np.ndarray
    mag: np.ndarray
    fs_env: float
    band: tuple[float, float]


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.999)
    if lo_n >= hi_n:
        raise ValueError(f"Invalid bandpass {lo}-{hi} at fs={fs}")
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x.astype(np.float64))


def envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(x.astype(np.float64)))


def spectral_kurtosis_band(
    x: np.ndarray,
    fs: float,
    f_lo: float = 1000.0,
    f_hi: float | None = None,
    n_bands: int = 8,
) -> tuple[float, float, float]:
    """Coarse kurtogram surrogate: pick band with max envelope kurtosis.

    Returns (lo, hi, kurtosis). Offline / calibration use — not the MCU path.
    """
    from scipy.stats import kurtosis

    f_hi = f_hi or min(0.45 * fs, 10000.0)
    edges = np.linspace(f_lo, f_hi, n_bands + 1)
    best = (f_lo, edges[1], -np.inf)
    for i in range(n_bands):
        lo, hi = float(edges[i]), float(edges[i + 1])
        try:
            xb = bandpass(x, fs, lo, hi)
        except ValueError:
            continue
        env = envelope(xb)
        k = float(kurtosis(env, fisher=True, bias=False))
        if k > best[2]:
            best = (lo, hi, k)
    return best


def compute_envelope_spectrum(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float] = (2000.0, 5000.0),
    decim: int = 16,
    target_seconds: float | None = None,
) -> EnvelopeSpectrum:
    """Bandpass → Hilbert → optional pad/trim → decimate → FFT of envelope."""
    lo, hi = band
    # clamp band to available Nyquist
    hi = min(hi, 0.45 * fs)
    lo = min(lo, hi * 0.5)
    xb = bandpass(x, fs, lo, hi)
    env = envelope(xb)

    if target_seconds is not None:
        n_need = int(round(target_seconds * fs))
        if len(env) < n_need:
            # tile to reach target length (synthetic / short windows)
            reps = int(np.ceil(n_need / len(env)))
            env = np.tile(env, reps)[:n_need]
        else:
            env = env[:n_need]

    # decimate for fine envelope FFT resolution
    if decim > 1 and len(env) >= decim * 8:
        env_d = decimate(env, decim, ftype="iir", zero_phase=True)
        fs_env = fs / decim
    else:
        env_d = env
        fs_env = fs

    n = len(env_d)
    w = windows.hann(n, sym=False)
    # remove mean (DC carrier)
    env_d = env_d - np.mean(env_d)
    mag = np.abs(np.fft.rfft(env_d * w)) * 2.0 / (np.sum(w) + 1e-12)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_env)
    return EnvelopeSpectrum(freqs=freqs, mag=mag, fs_env=fs_env, band=(lo, hi))


def envelope_resolution_hz(fs: float, seconds: float, decim: int = 16) -> float:
    """Envelope FFT bin width after decimation."""
    fs_env = fs / decim
    n = int(round(seconds * fs / decim))
    return fs_env / n if n > 0 else np.inf


def envelope_fault_features(
    env_spec: EnvelopeSpectrum,
    fr: float,
    geo: BearingGeometry,
    tol: float = 0.02,
    n_harmonics: int = 3,
    normalize: bool = True,
) -> dict[str, float]:
    """BPFO/BPFI/2×BSF/FTF harmonic energies + sideband pairs.

    If normalize, divide by total envelope spectral energy (ratio features).
    """
    ff = bearing_freqs(fr, geo)
    freqs, mag = env_spec.freqs, env_spec.mag
    total = float(np.sum(mag**2)) + 1e-12

    out: dict[str, float] = {}
    for name, fund in [
        ("BPFO", ff.bpfo),
        ("BPFI", ff.bpfi),
        ("2xBSF", ff.ball_fault),
        ("FTF", ff.ftf),
    ]:
        energies = harmonic_band_energies(freqs, mag, fund, n_harmonics, tol)
        for h, e in enumerate(energies, start=1):
            val = e / total if normalize else e
            out[f"env_{name}_h{h}"] = val

    # sidebands
    bpfi_sb = sideband_pair_energy(freqs, mag, ff.bpfi, fr, tol)
    ball_sb = sideband_pair_energy(freqs, mag, ff.ball_fault, ff.ftf, tol)
    # also BPFO±fr (usually weak) and 2xBSF±fr as extras folded into pair features
    out["env_BPFI_sideband_fr"] = bpfi_sb / total if normalize else bpfi_sb
    out["env_2xBSF_sideband_FTF"] = ball_sb / total if normalize else ball_sb
    # two more for the planned count of 4 sideband features
    out["env_BPFO_sideband_fr"] = (
        sideband_pair_energy(freqs, mag, ff.bpfo, fr, tol) / total if normalize else 0.0
    )
    out["env_BPFI_h2_sideband_fr"] = (
        sideband_pair_energy(freqs, mag, 2 * ff.bpfi, fr, tol) / total
        if normalize
        else 0.0
    )
    return out


ENVELOPE_FEATURE_NAMES = (
    [f"env_{n}_h{h}" for n in ("BPFO", "BPFI", "2xBSF", "FTF") for h in (1, 2, 3)]
    + [
        "env_BPFI_sideband_fr",
        "env_2xBSF_sideband_FTF",
        "env_BPFO_sideband_fr",
        "env_BPFI_h2_sideband_fr",
    ]
)
