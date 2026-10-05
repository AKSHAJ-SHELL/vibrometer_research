#!/usr/bin/env python3
"""GBDT seed stability of the honest-split headline: pooled macro-F1 over 5 seeds (P3).

Writes results/real/seed_stability.json. Logistic regression is deterministic (lbfgs), so only
GBDT is checked.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibedge.models import predict_folds  # noqa: E402
from vibedge.real_experiments import _cols, build_table  # noqa: E402
from vibedge.splits import make_splits  # noqa: E402

SEEDS = range(5)


def run(ds: str, fs: str, seed: int) -> tuple[str, str, int, float]:
    warnings.filterwarnings("ignore")
    t = build_table(ds)
    y = t.y
    sp = make_splits(3, **_cols(t))
    pred = np.empty_like(y)
    for te, p in predict_folds("gbdt", t.X[fs], y, sp, seed=seed):
        pred[te] = p
    idx = np.concatenate([s.test_idx for s in sp])
    return ds, fs, seed, float(f1_score(y[idx], pred[idx], average="macro", labels=sorted(set(y.tolist()))))


def main() -> int:
    jobs = [(ds, fs, s) for ds in ("cwru", "paderborn") for fs in ("time_only", "envelope_ratio") for s in SEEDS]
    res = Parallel(n_jobs=-1)(delayed(run)(*j) for j in jobs)
    out = {"model": "gbdt", "protocol": 3, "seeds": list(SEEDS), "results": {}}
    for ds, fs, seed, f in res:
        out["results"].setdefault(ds, {}).setdefault(fs, {})[str(seed)] = f
    for ds, d in out["results"].items():
        for fs, v in d.items():
            vals = np.array(list(v.values()))
            d[fs] = {"per_seed": v, "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                     "range": float(vals.max() - vals.min())}
            print(f"{ds:10s} {fs:15s} mean {vals.mean():.3f}  sd {vals.std(ddof=1):.4f}")
    path = ROOT / "results" / "real" / "seed_stability.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
