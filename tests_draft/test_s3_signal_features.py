"""Spec §3 — Signal features with known answers.

Targets: vibedge.features.time_domain, .spectral, .envelope
"""

import numpy as np
import pytest
from scipy.stats import kurtosis

from vibedge.features.envelope import compute_envelope_spectrum, envelope
from vibedge.features.spectral import single_sided_amplitude, spectral_energy
from vibedge.features.time_domain import KURTOSIS_CONVENTION, time_domain_features

TOL = dict(abs=1e-6)


@pytest.fixture(scope="module")
def unit_sine():
    fs = 100_000
    return np.sin(2 * np.pi * 50 * np.arange(fs) / fs)


@pytest.fixture(scope="module")
def sine_50hz_fs1000():
    fs, n = 1000.0, 4096
    return np.sin(2 * np.pi * 50 * np.arange(n) / fs), fs


@pytest.fixture(scope="module")
def am_signal():
    fs = 12_000.0
    t = np.arange(int(2 * fs)) / fs
    return (1 + 0.5 * np.cos(2 * np.pi * 100 * t)) * np.sin(2 * np.pi * 3000 * t), fs


def test_unit_sine_rms_and_crest(unit_sine):
    # 3.1
    f = time_domain_features(unit_sine)
    assert f["rms"] == pytest.approx(0.707107, **TOL)
    assert f["crest_factor"] == pytest.approx(1.414214, **TOL)


def test_kurtosis_convention(unit_sine):
    # 3.1 — features quote PEARSON; state it in the paper
    assert KURTOSIS_CONVENTION == "pearson"
    assert time_domain_features(unit_sine)["kurtosis"] == pytest.approx(1.5, **TOL)
    assert kurtosis(unit_sine) == pytest.approx(-1.5, **TOL)  # scipy default = Fisher
    assert kurtosis(unit_sine, fisher=False) == pytest.approx(1.5, **TOL)


def test_gaussian_noise_moments():
    # 3.2 — seed pinned
    x = np.random.default_rng(0).standard_normal(1_000_000)
    f = time_domain_features(x)
    assert f["rms"] == pytest.approx(1.0, abs=0.01)
    assert f["kurtosis"] == pytest.approx(3.0, abs=0.02)
    assert kurtosis(x) == pytest.approx(0.0, abs=0.02)


def test_fft_peak_bin_and_leakage_amplitude(sine_50hz_fs1000):
    # 3.3 — 50 Hz falls between bins: 0.935, not 1.0
    x, fs = sine_50hz_fs1000
    freqs, amp = single_sided_amplitude(x, fs)
    k = int(np.argmax(amp))
    assert k == 205
    assert freqs[1] == pytest.approx(0.244141, **TOL)
    assert amp[k] == pytest.approx(0.935489, abs=1e-4)


def test_parseval(sine_50hz_fs1000):
    # 3.4
    x, _ = sine_50hz_fs1000
    assert float(np.sum(x**2)) == pytest.approx(2048.0, rel=1e-9)
    assert spectral_energy(x) == pytest.approx(2048.0, rel=1e-9)


@pytest.mark.parametrize("path", ["raw", "pipeline_decim1", "pipeline_decim16"])
def test_envelope_recovers_modulation_frequency(am_signal, path):
    # 3.5 — if this fails, every envelope feature in the paper is wrong
    x, fs = am_signal
    if path == "raw":
        e = envelope(x)
        e = e - e.mean()
        mag = np.abs(np.fft.rfft(e))
        freqs = np.fft.rfftfreq(len(e), 1.0 / fs)
    else:
        es = compute_envelope_spectrum(x, fs, band=(2000.0, 5000.0), decim=1 if path.endswith("1") else 16)
        freqs, mag = es.freqs, es.mag
    assert freqs[int(np.argmax(mag))] == pytest.approx(100.0, **TOL)
    assert freqs[1] == pytest.approx(0.5, **TOL)
