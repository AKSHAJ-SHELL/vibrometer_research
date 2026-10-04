"""Spec §7 — Energy integration (INA219). Trapezoid, abs 0.01."""

import numpy as np
import pytest

from vibedge.energy import integrate_energy_j


def _step(dt):
    t = np.arange(0, 10 + dt / 2, dt)
    return t, np.where(t >= 5.0, 10.0, 0.0)


def test_constant_5w_for_10s():
    t = np.linspace(0, 10, 1001)
    assert integrate_energy_j(t, np.full_like(t, 5.0)) == pytest.approx(50.0, abs=0.01)


def test_ramp():
    t = np.linspace(0, 10, 1001)
    assert integrate_energy_j(t, t) == pytest.approx(50.0, abs=0.01)


def test_step_fine_sampling():
    # error ≈ ΔP·dt/2, so dt = 1e-4 → 50.0005 J
    assert integrate_energy_j(*_step(1e-4)) == pytest.approx(50.0, abs=0.01)


def test_coarse_grid_is_caught():
    # dt = 1 s smears the 10 W step over a second: 55 J
    assert abs(integrate_energy_j(*_step(1.0)) - 50.0) > 0.01
