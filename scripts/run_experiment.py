#!/usr/bin/env python3
"""Run experiments from config (C1 ladder by default).

  python scripts/run_experiment.py                       # synthetic → results/c1_results.json
  python scripts/run_experiment.py --dataset cwru        # real CWRU → results/c1_cwru.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibedge.experiments import _json_default, pooled_rows, run_c1_cwru, run_c1_synthetic, run_from_config


def _summary(result: dict) -> None:
    print(f"C1 summary — {result.get('dataset', '')} (pooled over folds):")
    print(f"  {'feature_set':16s} {'proto':>5s} {'folds':>5s} {'macroF1':>8s} {'acc':>6s} {'maj-F1':>7s}")
    for r in sorted(pooled_rows(result), key=lambda r: (r["protocol"], r["feature_set"])):
        print(
            f"  {r['feature_set']:16s} P{r['protocol']:<4d} {r['n_folds']:5d} "
            f"{r['macro_f1']:8.3f} {r['accuracy']:6.3f} {r['majority_baseline_f1']:7.3f}"
        )
    for n in result.get("notes", []):
        print(f"  note: {n}")
    print(f"Hypothesis: {result['hypothesis']}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    p.add_argument("--dataset", choices=["synthetic", "cwru"], default="synthetic")
    p.add_argument("--require-frozen", action="store_true")
    args = p.parse_args()

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "cwru":
        result = run_c1_cwru(
            root=ROOT / "data/cwru",
            freeze_dir=ROOT / "configs/frozen/cwru",
            require_frozen=args.require_frozen,
        )
        (out_dir / "c1_cwru.json").write_text(json.dumps(result, indent=2, default=_json_default))
        (out_dir / "cost_table_cwru.json").write_text(json.dumps(result["cost"], indent=2, default=_json_default))
    elif args.require_frozen:
        result = run_c1_synthetic(freeze_dir=ROOT / "configs/frozen", require_frozen=True)
        (out_dir / "c1_results.json").write_text(json.dumps(result, indent=2, default=_json_default))
        (out_dir / "cost_table.json").write_text(json.dumps(result["cost"], indent=2, default=_json_default))
    else:
        result = run_from_config(args.config)

    _summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
