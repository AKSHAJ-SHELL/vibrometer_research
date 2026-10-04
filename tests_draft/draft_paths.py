"""Shared paths and markers for the draft verifier (not named conftest to avoid clashing with tests/)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL_CWRU = ROOT / "data" / "cwru"

real_data = pytest.mark.skipif(not REAL_CWRU.exists(), reason="data/cwru not downloaded")
