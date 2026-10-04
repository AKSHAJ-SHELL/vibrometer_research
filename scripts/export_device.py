#!/usr/bin/env python3
"""Export the deployable models, test windows and expected outputs for the ESP32-S3.

Writes into firmware/vibedge_esp32s3/:
  vibedge_config.h   window length, filters (second-order sections + initial conditions), geometry
  vibedge_model.h    logistic regression per feature set (scaler + weights), trained on ALL CWRU windows
  vibedge_windows.h  8 real CWRU windows (2 per class, distinct physical faults) as int16 + scale + speed
  vibedge_golden.h   expected feature vectors and predicted classes from device_ref (float64)
and results/real/device_export.json (what was exported, honest-split accuracy at this window length,
and the gap between device_ref and the main pipeline).

Shaft speed: measured (from each file's RPM field) — the paper's headline mode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibedge.datasets.cwru import load_cwru_directory  # noqa: E402
from vibedge.device_ref import DECIM, SET_NAMES, TOL, device_features, make_filters  # noqa: E402
from vibedge.experiments import _fr_from_meta, _geo_from_meta, score_protocol  # noqa: E402
from vibedge.features import FEATURE_SET_GROUPS, extract_features  # noqa: E402
from vibedge.models import fit_model, model_param_count  # noqa: E402
from vibedge.splits import make_splits  # noqa: E402
from vibedge.taxonomy import CLASSES, CLASS_TO_IDX  # noqa: E402

N = 16384                 # power of two (radix-2 FFT on the device); 1.365 s at 12 kHz
FS = 12000.0
OUT = ROOT / "firmware" / "vibedge_esp32s3"
# 2 per class, each a different physical fault (load 0 recordings)
TEST_FILES = ["97.mat", "100.mat", "105.mat", "169.mat", "130.mat", "197.mat", "118.mat", "185.mat"]


def c_array(name: str, arr, ctype: str = "float", per_line: int = 8) -> str:
    flat = np.asarray(arr).ravel()

    def lit(v) -> str:
        if ctype.startswith(("int", "uint")):
            return str(int(v))
        t = f"{float(v):.17g}" if ctype == "double" else f"{float(v):.9g}"
        if not any(c in t for c in ".eEn"):     # 1 → 1.0 (a bare 1f is not valid C)
            t += ".0"
        return t + ("f" if ctype == "float" else "")

    body = ",\n  ".join(", ".join(lit(v) for v in flat[i:i + per_line]) for i in range(0, len(flat), per_line))
    shape = "".join(f"[{d}]" for d in np.shape(arr))
    return f"static const {ctype} {name}{shape} = {{\n  {body}\n}};\n"


def main() -> int:
    recs = load_cwru_directory(ROOT / "data" / "cwru", target_fs=FS)
    flt = make_filters(FS)

    # ---- all non-overlapping N-sample windows → training set (pipeline features) ----
    X = {s: [] for s in SET_NAMES}
    y, fids, recids, loads, sevs, labels = [], [], [], [], [], []
    for r in recs:
        geo, fr = _geo_from_meta(r.meta), _fr_from_meta(r.meta)
        for k in range(len(r.signal) // N):
            x = r.signal[k * N:(k + 1) * N]
            res = extract_features(x, FS, geo=geo, fr_hz=fr, slow_seconds=N / FS)
            for s in SET_NAMES:
                vec, names = res.subset(FEATURE_SET_GROUPS[s])
                assert names == SET_NAMES[s]
                X[s].append(vec)
            y.append(CLASS_TO_IDX[r.meta.label]); labels.append(r.meta.label)
            fids.append(r.meta.fault_id); recids.append(r.meta.file_id)
            loads.append(r.meta.load_hp); sevs.append(r.meta.severity)
    y = np.array(y)
    X = {s: np.nan_to_num(np.vstack(v)) for s, v in X.items()}
    label_universe = sorted(set(y.tolist()))

    # honest-split accuracy at THIS window length (what the deployed model is worth)
    cols = dict(recording_ids=recids, fault_ids=fids, labels=labels, loads=loads, severities=sevs,
                datasets=["cwru"] * len(y))
    honest = {}
    for p in (3, 4):
        sp = make_splits(p, **cols)
        for s in SET_NAMES:
            _, pooled = score_protocol(X[s], y, sp, label_universe)
            honest[f"P{p}|{s}"] = round(pooled["macro_f1"], 4)

    models = {s: fit_model("logreg", X[s], y) for s in SET_NAMES}

    # ---- test windows (first window of each chosen file), quantised to int16 ----
    by_name = {r.meta.file_id: r for r in recs}
    win_q, scales, frs, true_cls, gold, gold_pred, names_used = [], [], [], [], {s: [] for s in SET_NAMES}, {s: [] for s in SET_NAMES}, []
    pipeline_gap = {s: 0.0 for s in SET_NAMES}
    for fname in TEST_FILES:
        r = by_name[fname]
        x = r.signal[:N]
        scale = float(np.max(np.abs(x))) / 32767.0
        q = np.round(x / scale).astype(np.int16)
        xq = q.astype(np.float64) * scale            # what the device sees
        geo, fr = _geo_from_meta(r.meta), _fr_from_meta(r.meta)
        res = extract_features(xq, FS, geo=geo, fr_hz=fr, slow_seconds=N / FS)
        for s in SET_NAMES:
            ref = device_features(xq, fr, geo, flt, s)
            pipe, _ = res.subset(FEATURE_SET_GROUPS[s])
            pipeline_gap[s] = max(pipeline_gap[s], float(np.max(np.abs(ref - pipe) / np.maximum(np.abs(pipe), 1e-9))))
            gold[s].append(ref)
            gold_pred[s].append(int(models[s].predict(ref[None, :])[0]))
        win_q.append(q); scales.append(scale); frs.append(fr); true_cls.append(CLASS_TO_IDX[r.meta.label])
        names_used.append(fname)

    OUT.mkdir(parents=True, exist_ok=True)
    geo = _geo_from_meta(recs[0].meta)
    nsec_bp, nsec_dec = flt.bp_sos.shape[0], flt.dec_sos.shape[0]

    cfg = [
        "// Generated by scripts/export_device.py — do not edit by hand.\n#pragma once\n",
        f"#define VB_N {N}\n#define VB_LOG2N {int(np.log2(N))}\n#define VB_FS {FS:.1f}f\n#define VB_DECIM {DECIM}\n",
        f"#define VB_TOL {TOL}f\n#define VB_BP_NSEC {nsec_bp}\n#define VB_BP_PAD {flt.bp_pad}\n",
        f"#define VB_DEC_NSEC {nsec_dec}\n#define VB_DEC_PAD {flt.dec_pad}\n",
        f"// CWRU 6205 drive-end geometry\n#define VB_GEO_N {geo.n}\n#define VB_GEO_D {geo.d}f\n#define VB_GEO_PITCH {geo.D}f\n#define VB_GEO_PHI_DEG {geo.phi_deg}f\n",
        c_array("VB_BP_SOS", flt.bp_sos), c_array("VB_BP_ZI", flt.bp_zi),
        c_array("VB_DEC_SOS", flt.dec_sos), c_array("VB_DEC_ZI", flt.dec_zi),
    ]
    (OUT / "vibedge_config.h").write_text("\n".join(cfg))

    mdl = ["// Generated by scripts/export_device.py — do not edit by hand.\n#pragma once\n",
           "// Logistic regression per feature set, trained on all CWRU windows of VB_N samples.\n"]
    params = {}
    for tag, s in (("TIME", "time_only"), ("ENV", "envelope_ratio")):
        pipe = models[s].model
        sc, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
        params[s] = model_param_count(models[s])
        mdl += [f"#define VB_{tag}_NF {len(SET_NAMES[s])}\n#define VB_{tag}_NC {len(clf.classes_)}\n",
                c_array(f"VB_{tag}_CLASSES", clf.classes_, "int"),
                c_array(f"VB_{tag}_MEAN", sc.mean_), c_array(f"VB_{tag}_SCALE", sc.scale_),
                c_array(f"VB_{tag}_COEF", clf.coef_), c_array(f"VB_{tag}_INTERCEPT", clf.intercept_)]
    (OUT / "vibedge_model.h").write_text("\n".join(mdl))

    win = ["// Generated by scripts/export_device.py — do not edit by hand.\n#pragma once\n#include <stdint.h>\n",
           f"#define VB_NWIN {len(win_q)}\n",
           "// " + ", ".join(f"{n} ({CLASSES[c]})" for n, c in zip(names_used, true_cls)) + "\n",
           c_array("VB_WIN_SCALE", scales), c_array("VB_WIN_FR", frs), c_array("VB_WIN_TRUE", true_cls, "int"),
           c_array("VB_WIN", np.stack(win_q), "int16_t", per_line=16)]
    (OUT / "vibedge_windows.h").write_text("\n".join(win))

    gh = ["// Generated by scripts/export_device.py — expected outputs from src/vibedge/device_ref.py (float64).\n#pragma once\n"]
    for tag, s in (("TIME", "time_only"), ("ENV", "envelope_ratio")):
        gh += [c_array(f"VB_GOLD_{tag}", np.stack(gold[s]), "double"),
               c_array(f"VB_GOLD_{tag}_PRED", gold_pred[s], "int")]
    (OUT / "vibedge_golden.h").write_text("\n".join(gh))

    summary = {
        "window_samples": N, "fs_hz": FS, "window_s": N / FS, "speed_source": "measured (RPM field)",
        "train_windows": int(len(y)), "train_recordings": len(set(recids)), "fault_identities": len(set(fids)),
        "test_windows": [{"file": n, "class": CLASSES[c]} for n, c in zip(names_used, true_cls)],
        "honest_split_macro_f1_at_this_window_length": honest,
        "device_ref_vs_pipeline_max_rel_diff": pipeline_gap,
        "model_params": params,
        "feature_names": SET_NAMES,
        "golden_pred_matches_true": {s: int(sum(p == t for p, t in zip(gold_pred[s], true_cls))) for s in SET_NAMES},
        "note": "Deployed models are trained on ALL CWRU windows (incl. the 8 test windows), so test-window "
                "predictions check the port, not accuracy. Accuracy is the honest-split entry above.",
    }
    res_path = ROOT / "results" / "real" / "device_export.json"
    res_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("train_windows", "honest_split_macro_f1_at_this_window_length",
                                             "device_ref_vs_pipeline_max_rel_diff", "golden_pred_matches_true")}, indent=1))
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("vibedge_*.h"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
