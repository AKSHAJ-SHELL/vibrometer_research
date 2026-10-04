"""Spec §4 — CWRU 6205 drive-end bearing frequencies (CWRU published table)."""

import pytest

from vibedge.features.bearing import CWRU_6205_DE, bearing_freqs


def test_multipliers_of_shaft_frequency():
    # 4.1
    assert (CWRU_6205_DE.n, CWRU_6205_DE.d, CWRU_6205_DE.D, CWRU_6205_DE.phi_deg) == (9, 0.3126, 1.537, 0.0)
    ff = bearing_freqs(1.0, CWRU_6205_DE)
    assert ff.bpfo == pytest.approx(3.584776, abs=1e-6)
    assert ff.bpfi == pytest.approx(5.415224, abs=1e-6)
    assert ff.ftf == pytest.approx(0.398308, abs=1e-6)
    assert ff.bsf == pytest.approx(2.356722, abs=1e-6)


def test_frequencies_at_1797_rpm():
    # 4.2
    fr = 1797 / 60
    assert fr == pytest.approx(29.95, abs=1e-12)
    ff = bearing_freqs(fr, CWRU_6205_DE)
    assert ff.bpfo == pytest.approx(107.364, abs=1e-3)
    assert ff.bpfi == pytest.approx(162.186, abs=1e-3)
    assert ff.bpfo + ff.bpfi == pytest.approx(269.55, abs=1e-6)
    assert ff.bpfo + ff.bpfi == pytest.approx(9 * fr, abs=1e-9)
