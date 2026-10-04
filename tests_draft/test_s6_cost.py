"""Spec §6 — Cost column (EMC²)."""

import json

import numpy as np
import pytest
from draft_paths import ROOT

from vibedge.cost import cost_table
from vibedge.datasets.synthetic import SyntheticConfig, generate_signal
from vibedge.features import FEATURE_SET_GROUPS


@pytest.fixture(scope="module")
def table():
    x, _, meta = generate_signal("bearing_IR", SyntheticConfig(duration=2.0, seed=3))
    fr = meta.extras["fr_hz"]
    counts = {r["feature_set"]: r["n_features"] for r in cost_table(x, meta.fs, fr, repeats=1)}
    rng = np.random.default_rng(0)
    X_by_set = {s: rng.standard_normal((30, n)) for s, n in counts.items()}
    y = np.repeat(np.arange(3), 10)
    return cost_table(x, meta.fs, fr, X_by_set=X_by_set, y=y, repeats=2)


def test_every_feature_set_has_a_cost_row(table):
    assert {r["feature_set"] for r in table} == set(FEATURE_SET_GROUPS)
    for r in table:
        assert r["n_features"] > 0
        assert r["feature_bytes_float32"] == 4 * r["n_features"]
        assert r["params_logreg"]["total"] > 0
        assert r["extract_s_per_window"] > 0


def test_full_feature_count_is_40(table):
    full = next(r for r in table if r["feature_set"] == "full")
    assert full["n_features"] == 40
    assert full["feature_bytes_float32"] == 160
    meta = json.loads((ROOT / "results" / "features_synthetic.json").read_text())
    assert meta["n_features"] == 40, "feature count changed — cost table is stale"
