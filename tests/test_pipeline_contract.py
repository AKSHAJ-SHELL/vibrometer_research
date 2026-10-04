"""Pipeline contract on the written results (E).

IMPLEMENTED by the agent at your request (2026-10-03) — for you to check.
The agent also wrote the code under test, so these are not an independent verifier
until you have reviewed them.

These read results/real/*.json and results/cache/, so they skip when those are
absent (run `make real` first).

Read first (once): https://docs.pytest.org/en/stable/getting-started.html
Each test below lists the links needed to implement THAT test under "Read:".

What the files look like:
  results/real/c1_{cwru,paderborn}[_gbdt].json → {"rows": [...]}
      pooled rows have row["fold"] == "pooled"; row["protocol"] is an int or "3b";
      honest-split pooled rows carry row["ci95"] = {"low", "high", "point", ...}
  results/cache/{dataset}_{pristine|deploy}_{speed}_{key}.npz
      speed is one of "oracle", "estimated", "prior" (real_experiments.SPEED_MODES)
      (9 older files without a speed mode are still there — see E3)
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "real"
CACHE = ROOT / "results" / "cache"

needs_results = pytest.mark.skipif(not RESULTS.exists(), reason="results/real missing — run `make real`")
needs_cache = pytest.mark.skipif(not CACHE.exists(), reason="results/cache missing — run `make real`")

C1_FILES = ["c1_cwru.json", "c1_paderborn.json", "c1_cwru_gbdt.json", "c1_paderborn_gbdt.json"]
SPEED_TAGS = ("_oracle_", "_estimated_", "_prior_")


def _pooled_rows(name):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} missing — run `make real`")
    rows = json.loads(path.read_text())["rows"]
    pooled = [r for r in rows if r["fold"] == "pooled"]
    assert pooled, f"{name} has no pooled rows"
    return pooled


@needs_results
@pytest.mark.parametrize("name", C1_FILES)
def test_e1_honest_pooled_rows_have_ci_around_point(name):
    """E1: every pooled row with protocol != 0 has "ci95" and low ≤ macro_f1 ≤ high.

    Read:
      https://docs.pytest.org/en/stable/how-to/parametrize.html
      https://docs.pytest.org/en/stable/how-to/skipping.html
      https://docs.python.org/3/library/pathlib.html
      https://docs.python.org/3/library/json.html
      (json is not on your list; json.loads reads the result files)
    """
    honest = [r for r in _pooled_rows(name) if str(r["protocol"]) != "0"]
    assert honest, f"{name} has no honest-split pooled rows"
    for r in honest:
        where = f"{name} P{r['protocol']} {r['feature_set']}"
        assert "ci95" in r, f"{where}: no ci95"
        assert r["ci95"]["low"] <= r["macro_f1"] <= r["ci95"]["high"], f"{where}: point outside CI"


@needs_results
@pytest.mark.parametrize("name", C1_FILES)
def test_e2_p0_rows_have_no_ci(name):
    """E2: pooled rows with protocol == 0 have NO "ci95" (P0 is invalid by design).

    Read:
      https://docs.pytest.org/en/stable/how-to/parametrize.html
      https://docs.python.org/3/library/pathlib.html
      https://docs.python.org/3/library/json.html
    """
    p0 = [r for r in _pooled_rows(name) if str(r["protocol"]) == "0"]
    assert p0, f"{name} has no P0 pooled rows"
    for r in p0:
        assert "ci95" not in r, f"{name} P0 {r['feature_set']}: has a ci95"


@needs_cache
def test_e3_speed_mode_is_in_cache_filename():
    """E3: every cache file name contains one of "_oracle_", "_estimated_", "_prior_",
    so oracle and estimated features can never share a cache entry.

    Read:
      https://docs.pytest.org/en/stable/how-to/skipping.html
      https://docs.python.org/3/library/pathlib.html
      (pathlib: Path.glob and Path.name)
    """
    files = sorted(CACHE.glob("*.npz"))
    assert files, "results/cache has no .npz files"
    untagged = [p.name for p in files if not any(tag in p.name for tag in SPEED_TAGS)]
    assert untagged == [], f"cache files without a speed mode in the name: {untagged}"
