#!/usr/bin/env python3
"""Build feature matrices for a dataset and save to npz."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibedge.datasets.synthetic import generate_dataset
from vibedge.deployment_view import to_deployment_view
from vibedge.experiments import FEATURE_SET_GROUPS, _build_feature_matrix
from vibedge.taxonomy import SampleMeta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "mafaulda", "cwru"])
    p.add_argument("--out", type=Path, default=Path("results/features_synthetic.npz"))
    p.add_argument("--deployment-view", action="store_true")
    p.add_argument("--feature-set", default="full", choices=list(FEATURE_SET_GROUPS))
    p.add_argument("--n-per-class", type=int, default=4)
    args = p.parse_args()

    if args.dataset == "synthetic":
        rows = generate_dataset(n_per_class=args.n_per_class, duration=2.0)
        pairs = []
        for i, (x, _, m) in enumerate(rows):
            if args.deployment_view:
                y, fs = to_deployment_view(x, m.fs, rng=np.random.default_rng(i))
                m = SampleMeta(
                    dataset=m.dataset,
                    file_id=m.file_id,
                    fault_id=m.fault_id,
                    label=m.label,
                    severity=m.severity,
                    rpm=m.rpm,
                    fs=fs,
                    geometry=m.geometry,
                    extras=dict(m.extras),
                )
                x = y
            pairs.append((x, m))
    else:
        raise SystemExit(
            f"Dataset {args.dataset} requires local files under data/{args.dataset}/"
        )

    groups = FEATURE_SET_GROUPS[args.feature_set]
    X, y, fault_ids, labels, severities = _build_feature_matrix(pairs, groups)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        fault_ids=np.array(fault_ids),
        labels=np.array(labels),
        severities=np.array(severities, dtype=object),
    )
    meta = {
        "dataset": args.dataset,
        "feature_set": args.feature_set,
        "groups": groups,
        "n": int(len(y)),
        "n_features": int(X.shape[1]),
        "deployment_view": args.deployment_view,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.out} shape={X.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
