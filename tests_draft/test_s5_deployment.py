"""Spec §5 — Deployment view."""

import pytest

from vibedge.features.bearing import CWRU_6205_DE, bearing_freqs
from vibedge.features.envelope import envelope_resolution_hz


def test_envelope_resolution_at_deployment_rate():
    res = envelope_resolution_hz(26_700.0, 4.0, 16)
    assert res == pytest.approx(0.25, abs=1e-6)
    assert res < 0.5  # ±2% bands at ~100 Hz


def test_decimation_keeps_fault_band_below_nyquist():
    # resolution is 1/T regardless of decimation — decimation only sets Nyquist
    fs_env = 26_700.0 / 16
    fr = 1797 / 60
    assert 3 * bearing_freqs(fr, CWRU_6205_DE).bpfi + fr < fs_env / 2
