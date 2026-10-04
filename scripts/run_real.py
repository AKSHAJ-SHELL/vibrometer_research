#!/usr/bin/env python3
"""Train + evaluate on the downloaded datasets; write results/real/*.json.

  python scripts/run_real.py                 # everything (features cached in results/cache/)
  python scripts/run_real.py --only cwru     # one dataset's C1 ladder
  python scripts/run_real.py --refresh       # re-extract features
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibedge.experiments import _json_default  # noqa: E402
from vibedge.real_experiments import (  # noqa: E402
    build_table,
    gbdt_loss_curves,
    mafaulda_severity,
    paderborn_artificial_to_real,
    run_c1,
    speed_accuracy,
    transfer_matrix,
)

OUT = ROOT / "results" / "real"

# Protocols per dataset (see taxonomy.PROTOCOL_FEASIBILITY). P1 is skipped where
# there is one window per recording — it would equal P0 without the leakage.
C1_PROTOCOLS = {
    "cwru": [0, 1, 2, 3, 4],
    "paderborn": [0, 2, 3],
    "mafaulda": [0],
}


# CWRU has one healthy bearing: under P3/P4 its F1 is structurally 0, so these
# rows also report macro-F1 over the fault classes only (DESIGN.md).
CWRU_EXCL_HEALTHY = (3, 4)


def _write(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=_json_default))
    print(f"  wrote results/real/{name}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["cwru", "paderborn", "mafaulda", "transfer", "deploy", "gbdt", "curves", "speed", "cost"], default=None)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    def want(x: str) -> bool:
        return args.only in (None, x)

    t0 = time.time()
    tables = {}
    for ds in ("cwru", "paderborn", "mafaulda"):
        if want(ds) or (ds == "cwru" and want("deploy")):
            print(f"[{time.time() - t0:6.0f}s] features: {ds} (pristine)", flush=True)
            tables[ds] = build_table(ds, n_jobs=args.n_jobs, refresh=args.refresh)
            print(f"  {len(tables[ds].meta)} windows", flush=True)

    if want("cwru"):
        _write("c1_cwru.json", run_c1(tables["cwru"], C1_PROTOCOLS["cwru"],
                                      excl_healthy_protocols=CWRU_EXCL_HEALTHY))
    if want("paderborn"):
        t = tables["paderborn"]
        _write("c1_paderborn.json",
               run_c1(t, C1_PROTOCOLS["paderborn"], extra_splits={"3b": paderborn_artificial_to_real(t)}))
    if want("mafaulda"):
        _write("c1_mafaulda.json", run_c1(tables["mafaulda"], C1_PROTOCOLS["mafaulda"]))
        _write("severity_mafaulda.json", mafaulda_severity(tables["mafaulda"]))

    if want("deploy"):
        # the same P3 ladder through the deployment-sensor view, same windows as pristine
        from vibedge.real_experiments import PRISTINE

        out = {"protocol": 3}
        for ds in ("cwru", "paderborn"):
            if ds not in tables:
                tables[ds] = build_table(ds, n_jobs=args.n_jobs)
            print(f"[{time.time() - t0:6.0f}s] features: {ds} (deployment)", flush=True)
            td = build_table(ds, deployment=True, n_jobs=args.n_jobs, refresh=args.refresh,
                             settings=PRISTINE[ds])
            out[ds] = {"pristine": run_c1(tables[ds], [3], freeze=False),
                       "deployment": run_c1(td, [3], freeze=False)}
        _write("deployment.json", out)

    if want("gbdt"):
        # same ladders with GBDT (tier 2) — slower; files suffixed _gbdt
        for ds in ("cwru", "paderborn"):
            if ds not in tables:
                tables[ds] = build_table(ds, n_jobs=args.n_jobs)
            print(f"[{time.time() - t0:6.0f}s] gbdt C1: {ds}", flush=True)
            extra = {"3b": paderborn_artificial_to_real(tables[ds])} if ds == "paderborn" else None
            excl = CWRU_EXCL_HEALTHY if ds == "cwru" else ()
            _write(f"c1_{ds}_gbdt.json", run_c1(tables[ds], C1_PROTOCOLS[ds], model_name="gbdt",
                                                freeze=False, extra_splits=extra, excl_healthy_protocols=excl))

    if want("curves"):
        for ds in ("cwru", "paderborn"):
            if ds not in tables:
                tables[ds] = build_table(ds, n_jobs=args.n_jobs)
            print(f"[{time.time() - t0:6.0f}s] gbdt loss curves: {ds}", flush=True)
            _write(f"loss_curves_{ds}.json", gbdt_loss_curves(tables[ds], n_jobs=args.n_jobs))

    if want("speed"):
        # oracle vs estimated vs estimated-with-rated-speed-prior, honest splits only
        honest = {"cwru": [3, 4], "paderborn": [3]}
        out = {}
        for ds, protos in honest.items():
            out[ds] = {}
            for mode in ("oracle", "estimated", "prior"):
                print(f"[{time.time() - t0:6.0f}s] speed={mode}: {ds}", flush=True)
                t = build_table(ds, n_jobs=args.n_jobs, refresh=args.refresh, speed=mode)
                extra = {"3b": paderborn_artificial_to_real(t)} if ds == "paderborn" else None
                res = run_c1(t, protos, freeze=False, extra_splits=extra)
                res["speed_accuracy"] = speed_accuracy(t)
                out[ds][mode] = res
        _write("speed_ablation.json", out)

    if want("cost"):
        # EMC² cost column on a real window: features, float32 bytes, params, host time
        from vibedge.cost import cost_table
        from vibedge.datasets.cwru import load_cwru_directory
        from vibedge.experiments import _fr_from_meta, _geo_from_meta
        from vibedge.real_experiments import DATA

        if "cwru" not in tables:
            tables["cwru"] = build_table("cwru", n_jobs=args.n_jobs)
        rec = load_cwru_directory(DATA / "cwru", target_fs=12000.0)[0]
        x = rec.signal[: int(4.0 * rec.meta.fs)]
        rows = cost_table(x, rec.meta.fs, _fr_from_meta(rec.meta), X_by_set=tables["cwru"].X,
                          y=tables["cwru"].y, geo=_geo_from_meta(rec.meta), repeats=20)
        _write("cost.json", {"dataset": "cwru", "window_file": rec.meta.file_id, "window_s": 4.0,
                             "fs": rec.meta.fs, "models_fit_on": "all CWRU windows", "rows": rows})

    if want("transfer"):
        dep = {}
        for ds in ("cwru", "paderborn", "mfpt"):
            print(f"[{time.time() - t0:6.0f}s] features: {ds} (deployment, transfer)", flush=True)
            dep[ds] = build_table(ds, deployment=True, n_jobs=args.n_jobs, refresh=args.refresh)
        _write("transfer.json", {fs: transfer_matrix(dep, fs) for fs in ("time_only", "envelope_ratio")})

    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
