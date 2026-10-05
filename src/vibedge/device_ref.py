"""Reference of the exact algorithm the ESP32 firmware runs (float64 numpy).

The firmware (firmware/vibedge_esp32/) is a line-by-line port of THIS file,
not of the main pipeline. This file is in turn checked against
vibedge.features.extract_features (scripts/export_device.py reports the gap),
so the chain is: main pipeline ≈ device_ref ≈ firmware.

Differences from the main pipeline, all numerically equivalent:
  * band-pass filtfilt runs in second-order sections (scipy: transfer function);
    same filter, same odd-extension padding (27 samples) and initial conditions;
  * window length is a power of two (FFT on the device is radix-2).
Shaft speed is passed in (measured speed, the paper's headline mode).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, cheby1, sosfilt, sosfilt_zi, tf2sos

from vibedge.features.bearing import BearingGeometry, bearing_freqs
from vibedge.features.envelope import ENVELOPE_FEATURE_NAMES
from vibedge.features.spectral import BROADBAND_FEATURE_NAMES, ORDER_FEATURE_NAMES
from vibedge.features.time_domain import FEATURE_NAMES as TIME_NAMES

BAND = (2000.0, 5000.0)
DECIM = 16
TOL = 0.02
EPS = 1e-12
ORDERS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)

ENV_HARMONIC_NAMES = [n for n in ENVELOPE_FEATURE_NAMES if "sideband" not in n]
ENV_SIDEBAND_NAMES = [n for n in ENVELOPE_FEATURE_NAMES if "sideband" in n]
# Same column order as FeatureResult.subset(groups)
SET_NAMES = {
    "time_only": TIME_NAMES + BROADBAND_FEATURE_NAMES,
    "envelope_ratio": TIME_NAMES + ORDER_FEATURE_NAMES + ENV_HARMONIC_NAMES
    + ENV_SIDEBAND_NAMES + BROADBAND_FEATURE_NAMES,
}


@dataclass(frozen=True)
class Filters:
    """Everything the firmware needs to reproduce the filters, as plain arrays."""

    fs: float
    bp_sos: np.ndarray       # (n_sec, 6) band-pass, butter(4, band), from the same b, a as the pipeline
    bp_zi: np.ndarray        # (n_sec, 2) sosfilt_zi(bp_sos)
    bp_pad: int              # filtfilt padlen = 3 * max(len(a), len(b))
    dec_sos: np.ndarray      # (n_sec, 6) cheby1(8, 0.05, 0.8 / DECIM) — scipy.decimate's filter
    dec_zi: np.ndarray
    dec_pad: int             # sosfiltfilt padlen


def make_filters(fs: float) -> Filters:
    nyq = fs / 2.0
    hi = min(BAND[1], 0.45 * fs)
    lo = min(BAND[0], hi * 0.5)
    b, a = butter(4, [max(lo / nyq, 1e-6), min(hi / nyq, 0.999)], btype="band")
    bp_sos = tf2sos(b, a)
    dec_sos = cheby1(8, 0.05, 0.8 / DECIM, output="sos")
    ntaps = 2 * dec_sos.shape[0] + 1
    ntaps -= min(int((dec_sos[:, 2] == 0).sum()), int((dec_sos[:, 5] == 0).sum()))
    return Filters(fs, bp_sos, sosfilt_zi(bp_sos), 3 * max(len(a), len(b)),
                   dec_sos, sosfilt_zi(dec_sos), 3 * ntaps)


def _odd_ext(x: np.ndarray, n: int) -> np.ndarray:
    left = 2 * x[0] - x[n:0:-1]
    right = 2 * x[-1] - x[-2:-(n + 2):-1]
    return np.concatenate([left, x, right])


def filtfilt_sos(sos: np.ndarray, zi: np.ndarray, pad: int, x: np.ndarray) -> np.ndarray:
    """scipy filtfilt / sosfiltfilt semantics: odd extension, steady-state initial conditions."""
    ext = _odd_ext(x, pad)
    y, _ = sosfilt(sos, ext, zi=zi * ext[0])
    y, _ = sosfilt(sos, y[::-1], zi=zi * y[-1])
    return y[::-1][pad:-pad]


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)  # periodic (sym=False)


def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    w = _hann(len(x))
    mag = np.abs(np.fft.rfft(x * w)) * 2.0 / (np.sum(w) + EPS)
    return np.fft.rfftfreq(len(x), 1.0 / fs), mag


def time_features(x: np.ndarray) -> dict[str, float]:
    a = np.abs(x)
    rms = np.sqrt(np.mean(x * x))
    peak = a.max()
    mu = x.mean()
    d = x - mu
    m2, m3, m4 = np.mean(d ** 2), np.mean(d ** 3), np.mean(d ** 4)
    mean_abs = a.mean()
    msa = np.mean(np.sqrt(a)) ** 2
    return {
        "rms": rms, "peak": peak, "peak_to_peak": x.max() - x.min(), "std": np.sqrt(m2),
        "kurtosis": m4 / m2 ** 2, "skewness": m3 / m2 ** 1.5,
        "crest_factor": peak / (rms + EPS), "shape_factor": rms / (mean_abs + EPS),
        "impulse_factor": peak / (mean_abs + EPS), "clearance_factor": peak / (msa + EPS),
    }


def broadband(freqs: np.ndarray, mag: np.ndarray, fs: float) -> dict[str, float]:
    p = mag ** 2
    psum = p.sum() + EPS
    pn = p / psum
    nz = pn > 0
    csum = np.cumsum(p)
    ridx = int(np.searchsorted(csum, 0.85 * csum[-1]))

    def band_e(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return p[m].sum() / psum

    return {
        "spectral_centroid": float(np.sum(freqs * pn)),
        "spectral_entropy": float(-np.sum(pn[nz] * np.log2(pn[nz] + 1e-300))),
        "spectral_flatness": float(np.exp(np.mean(np.log(mag + EPS))) / (np.mean(mag) + EPS)),
        "spectral_rolloff": float(freqs[min(ridx, len(freqs) - 1)]),
        "band_energy_1_2k": band_e(1000, 2000),
        "band_energy_2_4k": band_e(2000, 4000),
        "band_energy_4_6k": band_e(4000, min(6000, fs / 2 - 1)),
    }


def order_features(freqs: np.ndarray, mag: np.ndarray, fr: float) -> dict[str, float]:
    total = np.sum(mag ** 2) + EPS
    out = {}
    for o, name in zip(ORDERS, ORDER_FEATURE_NAMES):
        c = o * fr
        m = (freqs >= c * (1 - TOL)) & (freqs <= c * (1 + TOL))
        out[name] = float(np.sum(mag[m] ** 2) / total)
    return out


def _band_energy(freqs: np.ndarray, spec: np.ndarray, center: float) -> float:
    if center <= 0:
        return 0.0
    m = (freqs >= center * (1 - TOL)) & (freqs <= center * (1 + TOL))
    if not m.any():
        return float(spec[int(np.argmin(np.abs(freqs - center)))] ** 2)
    return float(np.sum(spec[m] ** 2))


def envelope_spectrum(x: np.ndarray, flt: Filters) -> tuple[np.ndarray, np.ndarray]:
    xb = filtfilt_sos(flt.bp_sos, flt.bp_zi, flt.bp_pad, x)
    n = len(xb)
    X = np.fft.fft(xb)
    h = np.zeros(n)
    h[0] = h[n // 2] = 1.0
    h[1:n // 2] = 2.0
    env = np.abs(np.fft.ifft(X * h))
    env_d = filtfilt_sos(flt.dec_sos, flt.dec_zi, flt.dec_pad, env)[::DECIM]
    env_d = env_d - env_d.mean()
    return spectrum(env_d, flt.fs / DECIM)


def envelope_features(freqs: np.ndarray, mag: np.ndarray, fr: float, geo: BearingGeometry) -> dict[str, float]:
    ff = bearing_freqs(fr, geo)
    total = np.sum(mag ** 2) + EPS
    out = {}
    for name, fund in (("BPFO", ff.bpfo), ("BPFI", ff.bpfi), ("2xBSF", ff.ball_fault), ("FTF", ff.ftf)):
        for h in (1, 2, 3):
            out[f"env_{name}_h{h}"] = _band_energy(freqs, mag, fund * h) / total

    def sb(carrier, spacing):
        return (_band_energy(freqs, mag, carrier - spacing) + _band_energy(freqs, mag, carrier + spacing)) / total

    out["env_BPFI_sideband_fr"] = sb(ff.bpfi, fr)
    out["env_2xBSF_sideband_FTF"] = sb(ff.ball_fault, ff.ftf)
    out["env_BPFO_sideband_fr"] = sb(ff.bpfo, fr)
    out["env_BPFI_h2_sideband_fr"] = sb(2 * ff.bpfi, fr)
    return out


def device_features(x: np.ndarray, fr: float, geo: BearingGeometry, flt: Filters,
                    feature_set: str) -> np.ndarray:
    """Feature vector in SET_NAMES[feature_set] order."""
    x = np.asarray(x, dtype=np.float64)
    vals = dict(time_features(x))
    freqs, mag = spectrum(x, flt.fs)
    vals.update(broadband(freqs, mag, flt.fs))
    if feature_set == "envelope_ratio":
        vals.update(order_features(freqs, mag, fr))
        ef, em = envelope_spectrum(x, flt)
        vals.update(envelope_features(ef, em, fr, geo))
    return np.array([vals[n] for n in SET_NAMES[feature_set]], dtype=np.float64)
