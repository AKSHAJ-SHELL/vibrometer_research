"""Figures from real-data results (results/real/*.json) → figures/real/.

Fixed encodings across every figure:
  feature set → colour + marker (time_only, envelope_ratio, full), never cycled;
  train-majority baseline → dashed grey; headline numbers are POOLED over folds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from vibedge.viz import COLORS_3, apply_style, save_figure

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "real"
OUT = Path(os.environ.get("VIBEDGE_FIG_OUT", ROOT / "figures" / "real")).resolve()   # the paper build redirects this

PAPER_MODE = bool(os.environ.get("VIBEDGE_PAPER"))   # paper build: compact labels for print size
SETS = ["time_only", "envelope_ratio", "full"]
SET_COLOR = dict(zip(SETS, COLORS_3))
SET_MARKER = {"time_only": "o", "envelope_ratio": "s", "full": "^"}
INK = "0.15"
MUTED = "0.45"
BASE = "0.55"

PROTO_NAME = {
    "0": "P0 random\nwindows",
    "1": "P1 leave-\nrecording",
    "2": "P2 leave-\nload",
    "3": "P3 leave-\nfault-id",
    "3b": "P3b artificial\n→ real",
    "4": "P4 leave-\nseverity",
}
DATASET_NAME = {"cwru": "CWRU (12k DE)", "paderborn": "Paderborn", "mafaulda": "MaFaulDa", "mfpt": "MFPT"}
SHORT_CLASS = {"healthy": "healthy", "imbalance": "imbal.", "misalignment": "misalign.",
               "bearing_IR": "IR", "bearing_OR": "OR", "bearing_ball": "ball"}


def _load(name: str) -> dict[str, Any] | None:
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def _pooled(res: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in res["rows"] if r["fold"] == "pooled"]


def _pooled_value(res, protocol, fs, key="macro_f1"):
    for r in _pooled(res):
        if str(r["protocol"]) == str(protocol) and r["feature_set"] == fs:
            return r[key]
    return np.nan


def fig_ladder(model: str = "logreg") -> Path | None:
    """C1 on real data: pooled macro-F1 by protocol, one panel per dataset."""
    suffix = "" if model == "logreg" else f"_{model}"
    # MaFaulDa has only P0 here (P3 infeasible on one rig); its honest result is C11
    results = {ds: _load(f"c1_{ds}{suffix}.json") for ds in ("cwru", "paderborn")}
    results = {k: v for k, v in results.items() if v}
    if not results:
        return None
    apply_style()
    widths = [max(len({str(r["protocol"]) for r in _pooled(v)}), 1.6) for v in results.values()]
    fig, axes = plt.subplots(1, len(results), figsize=(3.0 * sum(widths) / 2 + 2, 4.4),
                             sharey=True, gridspec_kw={"width_ratios": widths}, squeeze=False)
    data = {}
    for ax, (ds, res) in zip(axes[0], results.items()):
        protos = sorted({str(r["protocol"]) for r in _pooled(res)}, key=lambda p: (p.rstrip("b"), p))
        x = np.arange(len(protos))
        base = [_pooled_value(res, p, SETS[0], "majority_baseline_f1") for p in protos]
        ax.plot(x, base, ls="--", color=BASE, lw=1.5, marker="x", ms=6, zorder=1)
        data[ds] = {"protocols": protos, "majority_baseline_f1": base}
        for fs in SETS:
            ys = [_pooled_value(res, p, fs) for p in protos]
            data[ds][fs] = ys
            # full ≈ envelope_ratio on real data: draw full thin underneath so both stay visible
            thin = fs == "full"
            ax.plot(x, ys, color=SET_COLOR[fs], marker=SET_MARKER[fs], ms=5 if thin else 8,
                    lw=1.2 if thin else 2.2, markeredgecolor="white", markeredgewidth=1.0,
                    zorder=2 if thin else 3)
        ax.set_xticks(x)
        ax.set_xticklabels([PROTO_NAME.get(p, p) for p in protos], fontsize=8)
        ax.set_xlim(-0.4, len(protos) - 0.6 + (0.9 if len(protos) > 1 else 0.6))
        ax.set_title(f"{DATASET_NAME[ds]}\n{res['n_recordings']} recordings · "
                     f"{res['n_fault_identities']} fault ids", fontsize=9, color=INK)
        # direct labels at the last protocol, text in ink; baseline included so labels never overlap
        last = len(protos) - 1
        gap = 0.10 if os.environ.get("VIBEDGE_PAPER") else 0.055
        ends = sorted([(data[ds][fs][last], fs.replace("_", " "), INK) for fs in SETS]
                      + [(base[last], "train-majority", MUTED)], reverse=True)
        placed: list[float] = []
        for yv, label, color in ends:
            yl = yv
            for q in placed:
                if abs(yl - q) < gap:
                    yl = q - gap
            placed.append(yl)
            ax.annotate(label, (last, yv), xytext=(last + 0.12, yl), fontsize=7.5,
                        color=color, va="center", annotation_clip=False)
    axes[0][0].set_ylabel("Pooled macro-F1")
    axes[0][0].set_ylim(0, 1.05)
    handles = [plt.Line2D([], [], color=SET_COLOR[s], marker=SET_MARKER[s], lw=2, label=s) for s in SETS]
    handles.append(plt.Line2D([], [], color=BASE, ls="--", marker="x", label="train-majority baseline"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    model_name = {"logreg": "logistic regression", "gbdt": "gradient-boosted trees"}.get(model, model)
    fig.suptitle(f"C1 · Pooled macro-F1 by split protocol ({model_name})",
                 y=1.09, fontsize=11, color=INK)
    return save_figure(fig, OUT, f"C1_ladder_real{suffix}", data, source="CWRU, Paderborn KAt")


def fig_confusion_cwru() -> Path | None:
    res = _load("c1_cwru.json")
    if not res:
        return None
    apply_style()
    labels = [SHORT_CLASS[c] for c in res["labels"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    data = {}
    for ax, fs in zip(axes, ["time_only", "envelope_ratio"]):
        row = next(r for r in _pooled(res) if str(r["protocol"]) == "3" and r["feature_set"] == fs)
        cm = np.asarray(row["confusion"], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=35 if PAPER_MODE else 0,
                      ha="right" if PAPER_MODE else "center")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.grid(False)
        ax.set_title(f"{fs} — macro-F1 {row['macro_f1']:.2f}", fontsize=10, color=INK)
        data[fs] = {"labels": res["labels"], "confusion": cm.tolist()}
    axes[0].set_ylabel("True")
    fig.suptitle("C2 · CWRU, leave-one-fault-identity-out (P3), pooled window counts; shade = row share",
                 fontsize=10, color=INK)
    return save_figure(fig, OUT, "C2_confusion_cwru_P3", data, source="CWRU 12k drive end + normal")


def fig_fold_spread() -> Path | None:
    """Per-fold accuracy under P3 — which held-out faults fail."""
    res = _load("c1_cwru.json")
    if not res:
        return None
    apply_style()
    folds = {}
    for r in res["rows"]:
        if str(r["protocol"]) == "3" and r["fold"] != "pooled" and r["feature_set"] in ("time_only", "envelope_ratio"):
            folds.setdefault(r["meta"]["held_fault"], {})[r["feature_set"]] = r["accuracy"]
    order = sorted(folds, key=lambda f: folds[f].get("envelope_ratio", 0))
    fig, ax = plt.subplots(figsize=(7, 0.34 * len(order) + 1.4))
    y = np.arange(len(order))
    a = [folds[f]["time_only"] for f in order]
    b = [folds[f]["envelope_ratio"] for f in order]
    ax.hlines(y, np.minimum(a, b), np.maximum(a, b), color="0.8", lw=2, zorder=1)
    # small vertical offset so equal scores don't hide one marker under the other
    for fs, vals, dy in (("time_only", a, -0.14), ("envelope_ratio", b, 0.14)):
        ax.scatter(vals, y + dy, s=64, color=SET_COLOR[fs], marker=SET_MARKER[fs], zorder=3,
                   edgecolor="white", linewidth=1.2, label=fs)
    ax.set_yticks(y, [f.replace("cwru_", "").replace("_", " ") for f in order], fontsize=8)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Held-out fault: fraction of its windows classified correctly")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("C6 · CWRU P3 per-fold accuracy (each row = one held-out physical fault)",
                 fontsize=10, color=INK)
    return save_figure(fig, OUT, "C6_fold_spread_cwru_P3", {"folds": folds}, source="CWRU 12k drive end + normal")


def fig_transfer() -> Path | None:
    res = _load("transfer.json")
    if not res:
        return None
    apply_style()
    sets = [s for s in ("time_only", "envelope_ratio") if s in res]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 4.2))
    axes = np.atleast_1d(axes)
    for ax, fs in zip(axes, sets):
        r = res[fs]
        M = np.asarray(r["scores"])
        names = [DATASET_NAME[d] for d in r["datasets"]]
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if M[i, j] > 0.55 else INK,
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Test dataset")
        ax.grid(False)
        ax.set_title(fs, fontsize=10, color=INK)
    axes[0].set_ylabel("Train dataset")
    diag = res[sets[0]].get("diagonal_protocol", {})
    leaky = [DATASET_NAME[d] for d, p in diag.items() if p != 3]
    note = (f"; {', '.join(leaky)} diagonal is leave-one-recording-out (too few fault ids — leaky)"
            if leaky else "")
    fig.suptitle("C10 · Cross-dataset macro-F1 (healthy / IR / OR, deployment view)\n"
                 f"off-diagonal = train on row, test on column · diagonal = leave-one-fault-out{note}",
                 fontsize=9.5, color=INK)
    return save_figure(fig, OUT, "C10_transfer_real", res, source="CWRU, Paderborn KAt, MFPT")


def fig_severity() -> Path | None:
    res = _load("severity_mafaulda.json")
    if not res:
        return None
    apply_style()
    designs = list(res["designs"])
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.26
    x = np.arange(len(designs))
    for k, fs in enumerate(SETS):
        vals = [res["designs"][d]["by_set"][fs]["macro_f1"] for d in designs]
        bars = ax.bar(x + (k - 1) * width, vals, width - 0.03, color=SET_COLOR[fs], label=fs, zorder=2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                    fontsize=8, color=INK)
    base = [res["designs"][d]["by_set"]["full"]["majority_baseline_f1"] for d in designs]
    for xi, b in zip(x, base):
        ax.hlines(b, xi - 1.5 * width, xi + 1.5 * width, color=BASE, ls="--", lw=1.5, zorder=3)
    labels = []
    for d in designs:
        dd = res["designs"][d]
        labels.append(f"{d}\ntrain {', '.join(f'{s:g}' for s in dd['train_severities_g'])} g\n"
                      f"test {', '.join(f'{s:g}' for s in dd['test_severities_g'])} g")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Macro-F1 (healthy vs imbalance)")
    from matplotlib.patches import Patch

    handles = [Patch(color=SET_COLOR[fs], label=fs) for fs in SETS]
    handles.append(plt.Line2D([], [], color=BASE, ls="--", label="train-majority baseline"))
    if PAPER_MODE:   # reserved strip above the bars, clear of the large tick labels
        ax.set_ylim(0, 1.5)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper center", ncol=2)
    else:
        ax.legend(handles=handles, fontsize=8, frameon=True, framealpha=0.9, loc="lower right")
    ax.set_title("C11 · MaFaulDa imbalance, held-out severity (P4)", fontsize=10, color=INK)
    return save_figure(fig, OUT, "C11_severity_mafaulda", res, source="MaFaulDa (UFRJ), licence not stated")


def fig_deployment() -> Path | None:
    """P3 pooled macro-F1, pristine vs deployment view, per dataset (dumbbells)."""
    res = _load("deployment.json")
    if not res:
        return None
    apply_style()
    dss = [d for d in ("cwru", "paderborn") if d in res]
    fig, axes = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 3.2), sharex=True, squeeze=False)
    data = {}
    for ax, ds in zip(axes[0], dss):
        y = np.arange(len(SETS))
        data[ds] = {}
        for i, fs in enumerate(SETS):
            a = _pooled_value(res[ds]["pristine"], 3, fs)
            b = _pooled_value(res[ds]["deployment"], 3, fs)
            data[ds][fs] = {"pristine": a, "deployment": b}
            ax.plot([a, b], [i, i], color="0.8", lw=2, zorder=1)
            ax.scatter([a], [i], s=70, facecolor="white", edgecolor=SET_COLOR[fs], linewidth=2,
                       zorder=3, marker=SET_MARKER[fs])
            ax.scatter([b], [i], s=70, color=SET_COLOR[fs], zorder=3, marker=SET_MARKER[fs],
                       edgecolor="white", linewidth=1.2)
            ax.text(max(a, b) + 0.03, i, f"{a:.2f} → {b:.2f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, SETS)
        ax.set_xlim(0, 1.15)
        fs_note = "12 kHz source" if ds == "cwru" else "64 kHz source"
        ax.set_title(f"{ds.upper() if ds == 'cwru' else DATASET_NAME[ds]} ({fs_note})", fontsize=9, color=INK)
        ax.set_xlabel("Pooled macro-F1, P3")
    fig.suptitle("Deployment view (6 kHz LPF → 26.7 kSPS + MEMS noise): hollow = pristine, filled = deployment",
                 fontsize=10, color=INK)
    return save_figure(fig, OUT, "deployment_degradation_real", data, source="CWRU, Paderborn KAt")


def fig_envelope_cwru() -> Path | None:
    """Real CWRU envelope spectra with kinematic fault lines — a physics sanity check."""
    from vibedge.datasets.cwru import load_cwru_directory
    from vibedge.features.bearing import CWRU_6205_DE, bearing_freqs
    from vibedge.features.envelope import compute_envelope_spectrum

    root = ROOT / "data" / "cwru"
    if not root.exists():
        return None
    recs = {r.meta.file_id: r for r in load_cwru_directory(root, target_fs=12000.0)}
    picks = [("97.mat", "healthy (97)"), ("105.mat", "IR 0.007\" (105)"),
             ("130.mat", "OR 0.007\" @6 (130)"), ("118.mat", "ball 0.007\" (118)")]
    apply_style()
    fig, axes = plt.subplots(len(picks), 1, figsize=(8, 7.5), sharex=True)
    data = {}
    for ax, (fname, title) in zip(axes, picks):
        r = recs[fname]
        fr = r.meta.rpm / 60.0
        ff = bearing_freqs(fr, CWRU_6205_DE)
        n = int(4.0 * r.meta.fs)
        es = compute_envelope_spectrum(r.signal[:n], r.meta.fs, band=(2000.0, 5000.0), decim=4)
        m = es.freqs <= 450
        ax.plot(es.freqs[m], es.mag[m], color=COLORS_3[0], lw=1)
        top = es.mag[m].max()
        # labels sit beside their lines: BPFO to the left, 2×BSF centred higher, BPFI to the right
        for name, f0, h, ha in (("BPFO", ff.bpfo, 1.04, "right"), ("2×BSF", ff.ball_fault, 1.30, "center"),
                                ("BPFI", ff.bpfi, 1.04, "left")):
            ax.axvline(f0, color="0.6", ls=":", lw=1, zorder=0)
            ax.text(f0, top * h, name, fontsize=7, color=MUTED, ha=ha, va="bottom")
        ax.set_ylim(0, top * 1.55)
        ax.set_ylabel("|E(f)|", fontsize=8)
        ax.set_title(f"{title} · fr {fr:.2f} Hz", fontsize=9, color=INK, loc="left")
        data[fname] = {"fr_hz": fr, "BPFO": ff.bpfo, "BPFI": ff.bpfi, "2xBSF": ff.ball_fault,
                       "peak_hz": float(es.freqs[m][np.argmax(es.mag[m][5:]) + 5])}
    axes[-1].set_xlabel("Envelope frequency (Hz); band 2–5 kHz, 4 s")
    fig.suptitle("A3 · CWRU envelope spectra: IR peaks at BPFI, OR at BPFO (+ harmonics); the ball fault is weak",
                 fontsize=10, color=INK)
    return save_figure(fig, OUT, "A3_envelope_cwru", data, source="CWRU 12k drive end + normal")


def fig_loss_curves() -> Path | None:
    """GBDT train vs held-out log-loss and held-out macro-F1 per boosting stage."""
    res = {ds: _load(f"loss_curves_{ds}.json") for ds in ("cwru", "paderborn")}
    res = {k: v for k, v in res.items() if v}
    if not res:
        return None
    apply_style()
    fig, axes = plt.subplots(len(res), 2, figsize=(11, 3.6 * len(res)), squeeze=False)
    style = {0: dict(ls="--", lw=1.6), 3: dict(ls="-", lw=2.2)}
    pname = {0: "P0 random windows", 3: "P3 leave-fault-out"}
    for row, (ds, r) in zip(axes, res.items()):
        ax_l, ax_f = row
        stages = np.arange(1, r["n_estimators"] + 1)
        for fs in ("time_only", "envelope_ratio"):
            tr = r["curves"][f"{fs}|P3"]["train_logloss"]
            ax_l.plot(stages, np.clip(tr, 1e-4, None), color=SET_COLOR[fs], ls=":", lw=1.2, alpha=0.8)
            for p in (0, 3):
                c = r["curves"][f"{fs}|P{p}"]
                ax_l.plot(stages, np.clip(c["heldout_logloss"], 1e-4, None), color=SET_COLOR[fs], **style[p])
                ax_f.plot(stages, c["heldout_macro_f1"], color=SET_COLOR[fs], **style[p])
                if p == 3 and c["best_stage_by_heldout_loss"]:
                    b = c["best_stage_by_heldout_loss"]
                    ax_l.scatter([b], [c["heldout_logloss"][b - 1]], color=SET_COLOR[fs],
                                 marker=SET_MARKER[fs], s=50, zorder=4, edgecolor="white")
        ax_l.set_yscale("log")
        ax_l.set_ylabel("log-loss (floored at 1e-4)")
        ax_f.set_ylabel("held-out macro-F1 (pooled)")
        ax_f.set_ylim(0, 1.05)
        unseen = r["curves"]["envelope_ratio|P3"]["n_folds_unseen_class"]
        ax_l.set_title(f"{DATASET_NAME[ds]}: log-loss" if PAPER_MODE
                       else f"{DATASET_NAME[ds]} — log-loss (marker = held-out P3 minimum)",
                       fontsize=8.5, color=INK, loc="left")
        if unseen:
            ax_l.text(0.99, 0.02, "P3 loss excludes the\nhealthy fold" if PAPER_MODE else
                      f"held-out P3 loss excludes {unseen} fold whose class\nis absent from training (healthy)",
                      transform=ax_l.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)
        ax_f.set_title(f"{DATASET_NAME[ds]}: held-out macro-F1" if PAPER_MODE
                       else f"{DATASET_NAME[ds]} — held-out macro-F1", fontsize=8.5, color=INK, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("boosting stage")
    handles = [plt.Line2D([], [], color=SET_COLOR[fs], lw=2, label=fs) for fs in ("time_only", "envelope_ratio")]
    handles += [plt.Line2D([], [], color=INK, ls=":", lw=1.2, label="train (P3 folds, mean)"),
                plt.Line2D([], [], color=INK, ls="--", lw=1.6, label=f"held-out, {pname[0]}"),
                plt.Line2D([], [], color=INK, ls="-", lw=2.2, label=f"held-out, {pname[3]}")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("GBDT loss curves: random splits keep improving; leave-fault-out overfits after a few stages",
                 y=1.04, fontsize=10.5, color=INK)
    return save_figure(fig, OUT, "loss_curves_gbdt", res, source="CWRU, Paderborn KAt")


def fig_speed_ablation() -> Path | None:
    """Honest-split macro-F1 (95% fault-level CI) under oracle / estimated / prior speed."""
    res = _load("speed_ablation.json")
    if not res:
        return None
    apply_style()
    panels = [(ds, str(p)) for ds in res for p in
              sorted({str(r["protocol"]) for r in res[ds]["oracle"]["rows"] if r["fold"] == "pooled"})]
    modes = ["oracle", "estimated", "prior"]
    mode_name = {"oracle": "measured\nspeed", "estimated": "estimated\n10–60 Hz", "prior": "estimated\nrated ±10%"}
    if PAPER_MODE:   # 2×2 at print size: four panels in a row are too cramped for a 7 in figure
        ncols = 2
        nrows = -(-len(panels) // ncols)
        fig, grid = plt.subplots(nrows, ncols, figsize=(7.2, 2.9 * nrows), sharey=True, squeeze=False)
    else:
        fig, grid = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 4.4), sharey=True, squeeze=False)
    axes = [list(grid.ravel())]
    data = {}
    for ax, (ds, p) in zip(axes[0], panels):
        x = np.arange(len(modes))
        for k, fs in enumerate(("time_only", "envelope_ratio")):
            ys, lo, hi = [], [], []
            for m in modes:
                r = next(r for r in res[ds][m]["rows"]
                         if r["fold"] == "pooled" and str(r["protocol"]) == p and r["feature_set"] == fs)
                ys.append(r["macro_f1"]); lo.append(r["ci95"]["low"]); hi.append(r["ci95"]["high"])
            xo = x + (k - 0.5) * 0.22
            ax.errorbar(xo, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], fmt=SET_MARKER[fs],
                        color=SET_COLOR[fs], ms=8, capsize=3, lw=1.5, mec="white", label=fs)
            data[f"{ds}|P{p}|{fs}"] = {"macro_f1": ys, "low": lo, "high": hi, "modes": modes}
        labels = []
        short = {"oracle": "measured", "estimated": "10–60 Hz", "prior": "rated ±10%"}
        for m in modes:
            acc = res[ds][m]["speed_accuracy"]["within_2pct"]
            labels.append(f"{short[m]}\n{acc:.0%}" if PAPER_MODE else f"{mode_name[m]}\n{acc:.0%} within 2%")
        ax.set_xticks(x, labels, fontsize=7.5)
        if PAPER_MODE:
            ax.set_xlabel("speed source (share within 2%)")
        ax.set_xlim(-0.5, len(modes) - 0.5)
        ax.set_title(f"{ds.upper() if ds == 'cwru' else DATASET_NAME[ds]} · P{p}" if PAPER_MODE else
                     f"{DATASET_NAME[ds]} · {PROTO_NAME.get(p, p).replace("-" + chr(10), "-").replace(chr(10), " ")}",
                     fontsize=9, color=INK)
    if PAPER_MODE:
        for i in range(0, len(axes[0]), 2):
            axes[0][i].set_ylabel("Macro-F1 (95% CI)")
    else:
        axes[0][0].set_ylabel("Pooled macro-F1 (95% fault-level CI)")
    axes[0][0].set_ylim(0, 1)
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Estimated speed keeps the envelope advantage on CWRU (rated-speed prior) but loses most of it on Paderborn",
                 fontsize=10, color=INK)
    return save_figure(fig, OUT, "speed_ablation", data, source="CWRU, Paderborn KAt")


def make_real_figures() -> list[Path]:
    out = []
    for fn in (fig_ladder, lambda: fig_ladder("gbdt"), fig_loss_curves, fig_confusion_cwru, fig_fold_spread, fig_transfer, fig_severity,
               fig_deployment, fig_envelope_cwru, fig_speed_ablation):
        p = fn()
        if p is not None:
            out.append(p)
    return out
