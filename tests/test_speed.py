"""Speed estimator (A), no-oracle-leak (B) and speed-accuracy report (D).

IMPLEMENTED by the agent at your request (2026-10-03) — for you to check.
The agent also wrote the code under test, so these are not an independent verifier
until you have reviewed them. Expected values come from how each input is built.

Read first (once): https://docs.pytest.org/en/stable/getting-started.html
Each test below lists the links needed to implement THAT test under "Read:".

Functions under test (see the docstrings in src/vibedge/ for details):
  vibedge.speed.estimate_speed(x, fs, search_hz=(10.0, 60.0), n_harmonics=5, prefer="harmonic")
      -> SpeedEstimate(fr_hz: float, confidence: float, method: str, spectrum_peak_hz)
  vibedge.features.extract_features(x, fs, geo=None, fr_hz=None,
      speed_confidence_threshold=0.35, ..., search_hz=(10.0, 60.0), groups=None)
      -> FeatureResult(.vector, .names, .speed, .used_speed_features: bool)
  vibedge.real_experiments._features_for_recording(x, meta, window_s, overlap, max_windows,
      deployment, slow_seconds, seed, speed="oracle")
      -> list of (meta_row: dict, vecs: dict[feature_set -> np.ndarray], names: dict)
  vibedge.real_experiments.speed_accuracy(t: FeatureTable, tol=0.02) -> dict with keys
      within_2pct, within_5pct, median_rel_error, locked_to_2x, locked_to_half, ...
  vibedge.real_experiments.FeatureTable(dataset, X, names, meta, settings)
      speed_accuracy reads meta[i]["fr_true_hz"], ["fr_used_hz"], ["speed_confidence"],
      ["used_speed_features"]

Expected values come from how each signal is BUILT, not from running the code.
A1 and A4 are expected to FAIL on the current speed.py — that is the point.
"""

import copy

import numpy as np
import pytest

from vibedge.datasets.synthetic import SyntheticConfig, generate_signal
from vibedge.features import extract_features
from vibedge.real_experiments import FeatureTable, _features_for_recording, speed_accuracy
from vibedge.speed import estimate_speed

# Shared constants for A (from the test spec)
FS = 12_000.0      # Hz
DURATION = 4.0     # s  → bin width 1/4 s = 0.25 Hz
FR = 29.95         # Hz, true shaft rate (1797 rpm)
REL_TOL = 0.02     # ±2% — the fault-band tolerance the features rely on
CONF_THRESHOLD = 0.35  # extract_features' default speed_confidence_threshold


def _t():
    return np.arange(int(DURATION * FS)) / FS


def _sin(f):
    return np.sin(2 * np.pi * f * _t())


def _harmonic_series():
    return sum(_sin(k * FR) / k for k in range(1, 6))


# ---------------------------------------------------------------------------
# A. Speed estimator
# ---------------------------------------------------------------------------

def test_a1_harmonic_series():
    """A1: x = Σ_{k=1..5} sin(2π·k·FR·t) / k  →  fr_hz within 2% of FR.

    Today: fails (returns ≈ 37.46 Hz, i.e. 1.25×).

    Read:
      https://docs.pytest.org/en/stable/reference/reference.html#pytest-approx
      (rel=REL_TOL is a relative tolerance; building x is plain NumPy)
    """
    est = estimate_speed(_harmonic_series(), FS)
    assert est.fr_hz == pytest.approx(FR, rel=REL_TOL)


def test_a2_second_harmonic_louder_than_first():
    """A2: x = 0.3·sin(2π·FR·t) + 1.0·sin(2π·2FR·t) + 0.2·sin(2π·3FR·t)
    →  fr_hz within 2% of FR (NOT 2·FR ≈ 59.9).

    Read:
      https://docs.pytest.org/en/stable/reference/reference.html#pytest-approx
    """
    x = 0.3 * _sin(FR) + 1.0 * _sin(2 * FR) + 0.2 * _sin(3 * FR)
    est = estimate_speed(x, FS)
    assert est.fr_hz == pytest.approx(FR, rel=REL_TOL)
    assert est.fr_hz != pytest.approx(2 * FR, rel=REL_TOL)


def test_a3_pure_sine():
    """A3: x = sin(2π·FR·t)  →  |fr_hz − FR| ≤ 0.25 Hz (one bin at 4 s).

    Read:
      https://docs.pytest.org/en/stable/how-to/assert.html
    """
    est = estimate_speed(_sin(FR), FS)
    assert abs(est.fr_hz - FR) <= 1 / DURATION


def test_a4_white_noise_has_low_confidence():
    """A4: x = np.random.default_rng(0).standard_normal(int(DURATION * FS))
    →  est.confidence < 0.35  AND  extract_features(x, FS).used_speed_features is False.

    Today: fails (confidence = 1.00, so the speed-free fallback never triggers).

    Read:
      https://numpy.org/doc/stable/reference/random/generator.html
      https://docs.pytest.org/en/stable/how-to/assert.html
      (Generator.standard_normal is on the default_rng page)
    """
    x = np.random.default_rng(0).standard_normal(int(DURATION * FS))
    est = estimate_speed(x, FS)
    assert est.confidence < CONF_THRESHOLD
    r = extract_features(x, FS)  # fr_hz=None → the estimator is used
    assert r.used_speed_features is False


def test_a5_search_band_is_respected():
    """A5: the A1 signal with search_hz=(26.0, 33.0)  →  26 ≤ fr_hz ≤ 33.

    Read:
      https://docs.pytest.org/en/stable/how-to/assert.html
    """
    est = estimate_speed(_harmonic_series(), FS, search_hz=(26.0, 33.0))
    assert 26.0 <= est.fr_hz <= 33.0


# ---------------------------------------------------------------------------
# B. Estimated mode must never read the metadata speed
# ---------------------------------------------------------------------------

@pytest.fixture
def recording():
    """One synthetic recording + its SampleMeta, for B1/B2.

    Suggested: vibedge.datasets.synthetic.generate_signal("bearing_OR",
    SyntheticConfig(duration=4.0, seed=0)) returns (x, tacho, meta).
    Note: generate_signal sets meta.extras["fr_hz"], which _fr_from_meta reads BEFORE
    meta.rpm — remove it (meta.extras.pop("fr_hz")) so meta.rpm is the only speed source.

    Read:
      https://docs.pytest.org/en/stable/how-to/fixtures.html
    """
    x, _tacho, meta = generate_signal("bearing_OR", SyntheticConfig(duration=4.0, seed=0))
    meta.extras.pop("fr_hz")
    assert meta.rpm and meta.rpm > 0  # meta.rpm is now the only speed source
    return x, meta


def _features(x, meta, speed):
    """Helper: one window's feature vector for feature set 'full'.

    _features_for_recording(x, meta, window_s=4.0, overlap=0.0, max_windows=1,
                            deployment=False, slow_seconds=4.0, seed=0, speed=speed)
    → take [0][1]["full"].

    Read: nothing extra — the call is spelled out above.
    """
    rows = _features_for_recording(x, meta, window_s=4.0, overlap=0.0, max_windows=1,
                                   deployment=False, slow_seconds=4.0, seed=0, speed=speed)
    assert len(rows) == 1
    return rows[0][1]["full"]


def _pair(meta):
    ok, bad = copy.deepcopy(meta), copy.deepcopy(meta)
    bad.rpm = 1.0  # junk speed: 1 rpm
    return ok, bad


def test_b1_estimated_mode_ignores_meta_rpm(recording):
    """B1: same window, meta.rpm correct vs meta.rpm = 1.0, speed="estimated"
    →  np.array_equal(vec_correct, vec_junk) is True.

    Read:
      https://docs.python.org/3/library/copy.html
      https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
      https://docs.pytest.org/en/stable/how-to/fixtures.html
    """
    x, meta = recording
    ok, bad = _pair(meta)
    assert np.array_equal(_features(x, ok, "estimated"), _features(x, bad, "estimated"))


def test_b2_oracle_mode_does_read_meta_rpm(recording):
    """B2: the same pair with speed="oracle"  →  vectors DIFFER.

    Proves B1 is not passing vacuously (i.e. rpm really changes the features).

    Read:
      https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
    """
    x, meta = recording
    ok, bad = _pair(meta)
    assert not np.array_equal(_features(x, ok, "oracle"), _features(x, bad, "oracle"))


# ---------------------------------------------------------------------------
# D. Speed-accuracy report
# ---------------------------------------------------------------------------

def test_d_speed_accuracy_known_table():
    """D: fr_true = [30, 30, 30, 30], fr_used = [30, 60, 30.3, 15]
    →  within_2pct == 0.5   (30 and 30.3)
       locked_to_2x == 0.25 (60)
       locked_to_half == 0.25 (15)

    Build a minimal FeatureTable: X={}, names={}, settings={}, and meta = a list of
    dicts with keys fr_true_hz, fr_used_hz, speed_confidence, used_speed_features.

    Read:
      https://docs.pytest.org/en/stable/reference/reference.html#pytest-approx
      (0.5 and 0.25 are floats — compare with pytest.approx)
    """
    true = [30.0, 30.0, 30.0, 30.0]
    used = [30.0, 60.0, 30.3, 15.0]
    meta = [{"fr_true_hz": t, "fr_used_hz": u, "speed_confidence": 1.0, "used_speed_features": True}
            for t, u in zip(true, used)]
    rep = speed_accuracy(FeatureTable("test", X={}, names={}, meta=meta, settings={}))
    assert rep["n_windows"] == 4
    assert rep["within_2pct"] == pytest.approx(0.5)
    assert rep["locked_to_2x"] == pytest.approx(0.25)
    assert rep["locked_to_half"] == pytest.approx(0.25)
