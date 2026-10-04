"""Fault-level bootstrap (C).

IMPLEMENTED by the agent at your request (2026-10-03) — for you to check.
The agent also wrote the code under test, so these are not an independent verifier
until you have reviewed them. Expected values come from how each input is built.

Read first (once): https://docs.pytest.org/en/stable/getting-started.html
Each test below lists the links needed to implement THAT test under "Read:".

Functions under test (vibedge.evaluate):
  bootstrap_indices(fault_ids, rng) -> np.ndarray of row indices
      Draws len(unique faults) faults with rng.choice(faults, size=..., replace=True)
      and returns their rows, KEEPING duplicates. `faults` is the sorted unique ids.
  bootstrap_ci_fault_level(y_true, y_pred, fault_ids, n_boot=1000, seed=0, alpha=0.05,
                           labels=None) -> {"point", "mean", "low", "high", "n_boot", "n_faults", ...}
  paired_bootstrap_diff(y_true, pred_a, pred_b, fault_ids, n_boot=1000, seed=0, alpha=0.05)
      -> {"point", "low", "high", "p_le_0", "n_boot"}
"""

import numpy as np
import pytest

from vibedge.evaluate import bootstrap_ci_fault_level, bootstrap_indices, paired_bootstrap_diff

N_BOOT = 200  # enough for the invariants below; keeps the tests fast


class StubRng:
    """Stands in for np.random.Generator in C1: .choice always returns a fixed draw.

    Read:
      https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html
      (match choice(a, size=None, replace=True) so bootstrap_indices can call it)
    """

    def __init__(self, draw):
        self.draw = draw

    def choice(self, a, size=None, replace=True):
        return np.asarray(self.draw)


def test_c1_duplicates_are_kept():
    """C1: fault_ids = ["a", "a", "b"], stub draw = ["a", "a", "b"]
    →  indices [0, 1, 0, 1, 2]  (fault "a" drawn twice → its rows twice).

    The old np.isin code would give [0, 1, 2].

    Read:
      https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
      https://docs.pytest.org/en/stable/how-to/assert.html
    """
    idx = bootstrap_indices(["a", "a", "b"], StubRng(["a", "a", "b"]))
    assert idx.tolist() == [0, 1, 0, 1, 2]


def test_c2_perfect_predictions():
    """C2: y_pred == y_true  →  point == low == high == 1.0.

    Read:
      https://docs.pytest.org/en/stable/reference/reference.html#pytest-approx
      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
      (f1_score: why average="macro" over the present classes gives 1.0 here)
    """
    y = np.array([0, 0, 1, 1, 2, 2])
    faults = ["f0", "f0", "f1", "f1", "f2", "f2"]
    ci = bootstrap_ci_fault_level(y, y.copy(), faults, n_boot=N_BOOT, seed=0)
    assert ci["point"] == pytest.approx(1.0)
    assert ci["low"] == pytest.approx(1.0)
    assert ci["high"] == pytest.approx(1.0)


def test_c3_single_fault_gives_degenerate_interval():
    """C3: every row has the same fault id  →  low == high == point (all draws identical).

    Read:
      https://docs.pytest.org/en/stable/reference/reference.html#pytest-approx
    """
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])  # imperfect, so the point is not trivially 1
    ci = bootstrap_ci_fault_level(y_true, y_pred, ["only"] * 6, n_boot=N_BOOT, seed=0)
    assert ci["low"] == pytest.approx(ci["point"])
    assert ci["high"] == pytest.approx(ci["point"])


def test_c4_identical_models_have_zero_difference():
    """C4: paired_bootstrap_diff with pred_a == pred_b
    →  point == low == high == 0.0 and p_le_0 == 1.0.

    Read:
      https://docs.pytest.org/en/stable/reference/reference.html#pytest-approx
    """
    y = np.array([0, 0, 1, 1, 0, 1])
    pred = np.array([0, 1, 1, 0, 0, 1])
    d = paired_bootstrap_diff(y, pred, pred.copy(), ["a", "a", "b", "b", "c", "c"], n_boot=N_BOOT, seed=0)
    assert d["point"] == pytest.approx(0.0)
    assert d["low"] == pytest.approx(0.0)
    assert d["high"] == pytest.approx(0.0)
    assert d["p_le_0"] == pytest.approx(1.0)


def test_c5_same_seed_same_interval():
    """C5: two calls with the same seed  →  identical low/high.

    Read:
      https://docs.pytest.org/en/stable/how-to/assert.html
      https://numpy.org/doc/stable/reference/random/generator.html
      (default_rng: why the same seed reproduces the same draws)
    """
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    pred = np.array([0, 1, 1, 0, 0, 1, 1, 1])
    faults = ["a", "a", "b", "b", "c", "c", "d", "d"]
    one = bootstrap_ci_fault_level(y, pred, faults, n_boot=N_BOOT, seed=7)
    two = bootstrap_ci_fault_level(y, pred, faults, n_boot=N_BOOT, seed=7)
    assert (one["low"], one["high"]) == (two["low"], two["high"])
