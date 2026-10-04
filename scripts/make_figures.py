#!/usr/bin/env python3
"""Synthetic figure suite (smoke path, no downloads). Real-data figures: scripts/make_real_figures.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibedge.figures import make_all_figures


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "figures")
    p.add_argument("--synthetic", action="store_true",
                   help="required: this script only draws synthetic figures")
    p.add_argument("--skip-c1", action="store_true")
    args = p.parse_args()
    if not args.synthetic:
        p.error("only synthetic figures are drawn here; pass --synthetic, "
                "or use scripts/make_real_figures.py for real data")
    paths = make_all_figures(args.out, run_c1=not args.skip_c1)
    for path in paths:
        print(path)
    print(f"Wrote {len(paths)} figures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
