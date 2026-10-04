#!/usr/bin/env python3
"""Figures from results/real/*.json → figures/real/. Run scripts/run_real.py first."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibedge.figures_real import make_real_figures  # noqa: E402

if __name__ == "__main__":
    for p in make_real_figures():
        print(p.relative_to(ROOT))
