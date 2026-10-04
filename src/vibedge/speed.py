"""Shaft-speed estimation from vibration (no tachometer at deploy time).

Harmonic-comb score over a search band (first three shaft harmonics against the
spectral noise floor), refined from the detected harmonic peaks. Confidence is
the weighted share of harmonics that clear a noise-derived detection threshold.
Rewritten 2026-10-03: the old version locked to 1.25×/2× and reported
confidence ≈ 1 on white noise (see adjustments_and_cliffs.md C1b).
MaFaulDa's tacho channel is validation ground truth, not a runtime input.

When confidence < threshold, callers should fall back to the speed-free
feature subset (kurtosis, crest, flatness, band-energy ratios).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import windows


@dataclass(frozen=True)
class SpeedEstimate:
    fr_hz: float
    confidence: float
    method: str
    spectrum_peak_hz: float | None = None

    @property
    def rpm(self) -> float:
        return self.fr_hz * 60.0


def _rfft_mag(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    w = windows.hann(n, sym=False)
    X = np.fft.rfft(x * w)
    mag = np.abs(X) * 2.0 / np.sum(w)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, mag


# Detection threshold for a harmonic line, in multiples of the spectral noise
# floor (median magnitude). For Gaussian noise a Hann-windowed bin magnitude is
# Rayleigh-distributed, so P(|X| > r·median) = 2^(−r²). r = 3.65 gives a
# per-bin false-alarm rate of 1e-4 — set from noise statistics, not tuned.
DETECT_RATIO = 3.65
# Confidence weights for harmonics 1×, 2×, 3× (the fundamental counts most).
HARMONIC_WEIGHTS = (0.5, 0.25, 0.25)


def _harmonic_weights(n_harmonics: int) -> tuple[float, ...]:
    if n_harmonics == len(HARMONIC_WEIGHTS):
        return HARMONIC_WEIGHTS
    w = [0.5] + [0.5 / max(n_harmonics - 1, 1)] * (n_harmonics - 1)
    return tuple(w[:n_harmonics])


def _noise_floor(mag: np.ndarray) -> float:
    return float(np.median(mag)) + 1e-30


def _peak_near(freqs: np.ndarray, mag: np.ndarray, target: float, half_width_bins: int = 1) -> tuple[float, float]:
    """(magnitude, frequency) of the largest bin within ±half_width_bins of target."""
    df = freqs[1] - freqs[0]
    i = int(round(target / df))
    lo, hi = max(i - half_width_bins, 0), min(i + half_width_bins + 1, len(mag))
    if lo >= hi:
        return 0.0, target
    j = lo + int(np.argmax(mag[lo:hi]))
    # parabolic interpolation of the peak frequency
    if 0 < j < len(mag) - 1:
        a, b, c = mag[j - 1], mag[j], mag[j + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        return float(b), float(freqs[j] + np.clip(off, -0.5, 0.5) * df)
    return float(mag[j]), float(freqs[j])


def harmonic_comb_score(
    freqs: np.ndarray,
    mag: np.ndarray,
    fr: float,
    n_harmonics: int = 3,
    floor: float | None = None,
) -> float:
    """Weighted log of harmonic-to-noise-floor ratio over fr, 2fr, 3fr (each clipped at 1).

    A candidate earns nothing at a harmonic where there is no line, so a
    sub-harmonic (fr/2: no line at 1×, 3×) or a super-harmonic (2fr: no line at
    4×, 6× when the machine shows 1×–3×) scores below the true rate. Harmonics
    are weighted 0.5/0.25/0.25 (HARMONIC_WEIGHTS) so that, when only one line is
    visible, the candidate that explains it as its FUNDAMENTAL wins — otherwise
    fr, fr/2 and fr/3 would tie. No preference for higher or lower candidates.
    """
    if fr <= 0:
        return 0.0
    floor = floor if floor is not None else _noise_floor(mag)
    weights = _harmonic_weights(n_harmonics)
    total = 0.0
    for h in range(1, n_harmonics + 1):
        if h * fr >= freqs[-1]:
            break
        a, _ = _peak_near(freqs, mag, h * fr)
        total += weights[h - 1] * np.log(max(a / floor, 1.0))
    return total


def estimate_speed_harmonic(
    x: np.ndarray,
    fs: float,
    search_hz: tuple[float, float] = (10.0, 60.0),
    n_harmonics: int = 3,
) -> SpeedEstimate:
    """Harmonic-comb grid search, then refinement from the detected harmonic peaks.

    1. Spectrum (Hann) up to (n_harmonics + 1) × the top of the search band;
       noise floor = median magnitude there.
    2. Score every candidate on a half-bin grid inside search_hz
       (harmonic_comb_score); take the best.
    3. Refine: average peak_freq_h / h over the detected harmonics, weighted by
       magnitude; clip to search_hz.
    4. Confidence = weighted count of harmonics 1×–3× whose line exceeds
       DETECT_RATIO × noise floor (weights 0.5/0.25/0.25). White noise → ~0, so
       callers fall back to speed-free features below their threshold.
    """
    x = np.asarray(x, dtype=np.float64)
    freqs, mag = _rfft_mag(x - np.mean(x), fs)
    band = freqs <= (n_harmonics + 1) * search_hz[1]
    freqs_b, mag_b = freqs[band], mag[band]
    if len(freqs_b) < 8:
        return SpeedEstimate(fr_hz=0.0, confidence=0.0, method="harmonic_comb")
    floor = _noise_floor(mag_b[freqs_b > 0.5 * search_hz[0]])
    df = freqs_b[1] - freqs_b[0]

    grid = np.arange(search_hz[0], search_hz[1] + 1e-9, df / 2)
    scores = np.array([harmonic_comb_score(freqs_b, mag_b, f, n_harmonics, floor) for f in grid])
    best = float(grid[int(np.argmax(scores))])

    peaks = [_peak_near(freqs_b, mag_b, h * best) for h in range(1, n_harmonics + 1)]
    detected = [(h, a, f) for h, (a, f) in enumerate(peaks, start=1) if a > DETECT_RATIO * floor]
    if detected:
        w = np.array([a for _, a, _ in detected])
        est = float(np.sum(w * np.array([f / h for h, _, f in detected])) / np.sum(w))
    else:
        est = best
    est = float(np.clip(est, search_hz[0], search_hz[1]))

    weights = _harmonic_weights(n_harmonics)
    conf = float(sum(weights[h - 1] for h, _, _ in detected))
    return SpeedEstimate(fr_hz=est, confidence=conf, method="harmonic_comb",
                         spectrum_peak_hz=peaks[0][1] if peaks else None)


def estimate_speed_cepstrum(
    x: np.ndarray,
    fs: float,
    search_hz: tuple[float, float] = (10.0, 60.0),
) -> SpeedEstimate:
    """Cepstral quefrency peak → shaft period."""
    n = len(x)
    w = windows.hann(n, sym=False)
    spec = np.abs(np.fft.rfft(x * w)) + 1e-12
    log_spec = np.log(spec)
    cep = np.fft.irfft(log_spec)
    # quefrency in seconds: k / fs
    q = np.arange(len(cep)) / fs
    # search period band corresponding to search_hz
    t_lo = 1.0 / search_hz[1]
    t_hi = 1.0 / search_hz[0]
    mask = (q >= t_lo) & (q <= t_hi)
    if not np.any(mask):
        return SpeedEstimate(fr_hz=0.0, confidence=0.0, method="cepstrum")
    region = np.abs(cep[mask])
    q_region = q[mask]
    i = int(np.argmax(region))
    period = float(q_region[i])
    fr = 1.0 / period
    # confidence from peak prominence
    med = float(np.median(region)) + 1e-12
    conf = float(np.clip(np.tanh((region[i] - med) / (med * 2)), 0, 1))
    return SpeedEstimate(fr_hz=fr, confidence=conf, method="cepstrum")


def estimate_speed(
    x: np.ndarray,
    fs: float,
    search_hz: tuple[float, float] = (10.0, 60.0),
    n_harmonics: int = 3,
    prefer: str = "harmonic",
) -> SpeedEstimate:
    """Primary estimator: harmonic comb, with cepstrum as optional blend."""
    h = estimate_speed_harmonic(x, fs, search_hz, n_harmonics)
    if prefer == "cepstrum":
        return estimate_speed_cepstrum(x, fs, search_hz)
    if prefer == "blend":
        c = estimate_speed_cepstrum(x, fs, search_hz)
        # pick higher confidence; if close, average fr
        if abs(h.fr_hz - c.fr_hz) < 1.0 and min(h.confidence, c.confidence) > 0.2:
            fr = 0.5 * (h.fr_hz + c.fr_hz)
            conf = 0.5 * (h.confidence + c.confidence)
            return SpeedEstimate(fr_hz=fr, confidence=conf, method="blend")
        return h if h.confidence >= c.confidence else c
    return h


# Speed-free feature names (fallback when confidence is low)
SPEED_FREE_FEATURES: list[str] = [
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
    "spectral_centroid",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_rolloff",
    "band_energy_1_2k",
    "band_energy_2_4k",
    "band_energy_4_6k",
]
