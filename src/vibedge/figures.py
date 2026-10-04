"""Figure suite (~16 core figures). All run on synthetic data out of the box."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from vibedge.datasets.synthetic import generate_dataset, generate_signal, SyntheticConfig
from vibedge.features import extract_features
from vibedge.features.envelope import compute_envelope_spectrum
from vibedge.features.spectral import rfft_mag
from vibedge.taxonomy import CLASSES
from vibedge.viz import (
    CLASS_ORDER,
    COLORS_3,
    COLORS_6,
    PROTOCOL_LABELS,
    apply_style,
    save_figure,
)


def fig_a2_raw_fft(out_dir: Path) -> Path:
    """Raw FFT — why fault freqs are invisible under 1×."""
    apply_style()
    faults = ["healthy", "imbalance", "bearing_OR", "bearing_IR"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    data = {}
    for ax, fault, c in zip(axes.ravel(), faults, COLORS_6):
        x, _, meta = generate_signal(fault, SyntheticConfig(duration=1.0, seed=1))  # type: ignore[arg-type]
        freqs, mag = rfft_mag(x, meta.fs)
        m = freqs <= 500
        ax.plot(freqs[m], mag[m], color=c, lw=0.8)
        ax.set_title(fault)
        ax.set_ylabel("|X|")
        data[fault] = {"freqs": freqs[m][::10], "mag": mag[m][::10]}
    axes[-1, 0].set_xlabel("Hz")
    axes[-1, 1].set_xlabel("Hz")
    fig.suptitle("A2 Raw FFT — fault rates buried under 1× / harmonics")
    return save_figure(fig, out_dir, "A2_raw_fft", data)


def fig_a3_envelope(out_dir: Path) -> Path:
    apply_style()
    faults = ["healthy", "imbalance", "bearing_OR", "bearing_IR"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    data = {}
    for ax, fault, c in zip(axes.ravel(), faults, COLORS_6):
        x, _, meta = generate_signal(fault, SyntheticConfig(duration=4.0, seed=2))  # type: ignore[arg-type]
        env = compute_envelope_spectrum(x, meta.fs, target_seconds=4.0)
        m = env.freqs <= 300
        ax.plot(env.freqs[m], env.mag[m], color=c, lw=0.8)
        ax.set_title(fault)
        data[fault] = {"freqs": env.freqs[m][::5].tolist(), "mag": env.mag[m][::5].tolist()}
    fig.suptitle("A3 Envelope spectrum — bearing rates become discrete peaks")
    return save_figure(fig, out_dir, "A3_envelope_spectrum", data)


def fig_a4_side_by_side(out_dir: Path) -> Path:
    apply_style()
    x, _, meta = generate_signal("bearing_OR", SyntheticConfig(duration=4.0, seed=3))
    freqs, mag = rfft_mag(x, meta.fs)
    env = compute_envelope_spectrum(x, meta.fs, target_seconds=4.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    m1 = freqs <= 500
    ax1.plot(freqs[m1], mag[m1], color=COLORS_3[0], lw=0.8)
    ax1.set_title("Raw FFT")
    ax1.set_xlabel("Hz")
    m2 = env.freqs <= 300
    ax2.plot(env.freqs[m2], env.mag[m2], color=COLORS_3[1], lw=0.8)
    ax2.set_title("Envelope spectrum")
    ax2.set_xlabel("Hz")
    fig.suptitle("A4 Raw vs envelope (bearing OR) — the persuasive figure")
    return save_figure(
        fig,
        out_dir,
        "A4_raw_vs_envelope",
        {"raw_peak": float(mag[m1].max()), "env_peak": float(env.mag[m2].max())},
    )


def fig_a10_imbalance_dose(out_dir: Path) -> Path:
    """1× amplitude vs added mass (synthetic proxy for MaFaulDa severities)."""
    apply_style()
    sevs = [6, 10, 15, 20, 25, 30, 35]
    amps = []
    for s in sevs:
        x, _, meta = generate_signal(
            "imbalance",
            SyntheticConfig(duration=1.0, seed=s),
            severity=s / 10.0,
        )
        freqs, mag = rfft_mag(x, meta.fs)
        fr = meta.extras["fr_hz"]
        idx = int(np.argmin(np.abs(freqs - fr)))
        amps.append(float(mag[idx]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sevs, amps, "o-", color=COLORS_3[0])
    ax.set_xlabel("Imbalance severity (g proxy)")
    ax.set_ylabel("1× amplitude")
    ax.set_title("A10 Imbalance dose-response (synthetic)")
    return save_figure(fig, out_dir, "A10_imbalance_dose", {"severities": sevs, "amps": amps})


def fig_b6_leakage_diagnostic(out_dir: Path) -> Path:
    """t-SNE / PCA coloured by class then by file — if clusters by file, leakage."""
    apply_style()
    from sklearn.decomposition import PCA

    rows = generate_dataset(n_per_class=3, duration=2.0, base_seed=10)
    X, y, files = [], [], []
    for x, _, meta in rows:
        fr = extract_features(x, meta.fs, fr_hz=meta.extras["fr_hz"])
        X.append(fr.vector)
        y.append(meta.label)
        files.append(meta.file_id)
    X = np.nan_to_num(np.vstack(X), nan=0.0, posinf=0.0, neginf=0.0)
    # pad / trim to same length already ensured
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for i, cls in enumerate(CLASS_ORDER):
        m = np.array(y) == cls
        if not m.any():
            continue
        ax1.scatter(Z[m, 0], Z[m, 1], s=20, color=COLORS_6[i % 6], label=cls)
    ax1.legend(fontsize=7, loc="best")
    ax1.set_title("Coloured by class")
    # colour by file hash
    file_codes = np.array([hash(f) % 20 for f in files])
    sc = ax2.scatter(Z[:, 0], Z[:, 1], c=file_codes, s=20, cmap="tab20")
    ax2.set_title("Coloured by file (leakage diagnostic)")
    fig.suptitle("B6 Embedding — clustering by file ⇒ leakage")
    return save_figure(fig, out_dir, "B6_leakage_diagnostic", {"explained_var": pca.explained_variance_ratio_.tolist()})


def fig_c1_protocol_ladder(
    out_dir: Path,
    results: dict[str, Any] | None = None,
) -> Path:
    """Money figure: macro-F1 vs protocol × feature set, with majority baseline."""
    apply_style()
    if results is None:
        # run lightweight synthetic C1 inline
        from vibedge.experiments import run_c1_synthetic

        results = run_c1_synthetic()

    # headline = pooled rows (one per feature set × protocol); per-fold rows are diagnostics
    pooled = [r for r in results["rows"] if r.get("fold") == "pooled"]
    protocols = sorted({int(r["protocol"]) for r in pooled})
    feature_sets = sorted({r["feature_set"] for r in pooled})
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, fs_name in enumerate(feature_sets):
        ys = []
        for p in protocols:
            vals = [r["macro_f1"] for r in pooled if r["protocol"] == p and r["feature_set"] == fs_name]
            ys.append(float(vals[0]) if vals else np.nan)
        ax.plot(protocols, ys, marker="o", label=fs_name, color=COLORS_3[i % 3])
    # train-majority baseline, pooled per protocol (identical across feature sets)
    base = []
    for p in protocols:
        vals = [r["majority_baseline_f1"] for r in pooled if r["protocol"] == p]
        base.append(float(vals[0]) if vals else np.nan)
    ax.plot(protocols, base, ls="--", marker="x", color="0.4", label="train-majority baseline")
    ax.set_xticks(protocols)
    ax.set_xticklabels([PROTOCOL_LABELS.get(p, str(p)) for p in protocols], rotation=20, ha="right")
    ax.set_ylabel("Pooled macro-F1")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("C1 Feature set × protocol ladder (pooled over folds)")
    return save_figure(fig, out_dir, "C1_protocol_ladder", results)


def fig_c2_confusion(out_dir: Path, cm: np.ndarray, labels: list[str] | None = None) -> Path:
    apply_style()
    labels = labels or CLASS_ORDER[: cm.shape[0]]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("C2 Confusion (raw counts)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return save_figure(fig, out_dir, "C2_confusion", {"cm": cm, "labels": labels})


def fig_c6_lofo_spread(out_dir: Path, fold_scores: list[float]) -> Path:
    """Per-fold ACCURACY spread under leave-one-fault-out.

    Per-fold macro-F1 is undefined for single-class folds, so the spread is
    shown in accuracy (fraction of the held-out fault's windows classified
    correctly).
    """
    apply_style()
    if not fold_scores:
        raise ValueError("fold_scores required — no placeholder data")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(fold_scores, vert=True, widths=0.4)
    ax.scatter(np.ones(len(fold_scores)), fold_scores, color=COLORS_3[1], zorder=3)
    ax.set_ylabel("Per-fold accuracy")
    ax.set_title(f"C6 Leave-one-fault-out spread (n={len(fold_scores)} folds)")
    ax.set_xticks([])
    return save_figure(fig, out_dir, "C6_lofo_spread", {"fold_scores": fold_scores})


def fig_c10_transfer(out_dir: Path, matrix: dict) -> Path:
    apply_style()
    # real transfer matrices come from figures_real.fig_transfer; no placeholder data
    datasets = matrix["datasets"]
    M = np.asarray(matrix["scores"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(datasets)))
    ax.set_yticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Test →")
    ax.set_ylabel("Train ↓")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
    ax.set_title("C10 Cross-dataset transfer (macro-F1)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return save_figure(fig, out_dir, "C10_transfer", {"datasets": datasets, "scores": M})


def fig_c11_severity(out_dir: Path) -> Path:
    apply_style()
    # synthetic held-out severity demo
    from vibedge.experiments import run_severity_holdout_synthetic

    res = run_severity_holdout_synthetic()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["interp", "extrap"], [res["interp_f1"], res["extrap_f1"]], color=COLORS_3[:2])
    ax.axhline(res["majority_f1"], ls="--", color="0.4", label="majority")
    ax.set_ylabel("Macro-F1")
    ax.set_title("C11 Held-out severity (imbalance — honest protocol)")
    ax.legend()
    return save_figure(fig, out_dir, "C11_heldout_severity", res)


def fig_speed_validation(out_dir: Path) -> Path:
    apply_style()
    from vibedge.speed import estimate_speed
    from vibedge.datasets.synthetic import fr_from_tacho

    errs = []
    for fr in np.linspace(15, 50, 12):
        x, tacho, meta = generate_signal(
            "imbalance", SyntheticConfig(duration=2.0, fr_hz=float(fr), seed=int(fr * 10))
        )
        true_fr = fr_from_tacho(tacho, meta.fs)
        est = estimate_speed(x, meta.fs, search_hz=(10.0, 60.0))
        errs.append(est.fr_hz - true_fr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errs, bins=10, color=COLORS_3[0], edgecolor="white")
    ax.set_xlabel("Estimated − true f_r (Hz)")
    ax.set_title("Speed estimator error (synthetic tacho GT)")
    return save_figure(fig, out_dir, "speed_estimator_error", {"errors_hz": errs})


def SampleMeta_clone(m, fs):
    from vibedge.taxonomy import SampleMeta

    return SampleMeta(
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


def make_all_figures(out_dir: str | Path = "figures", run_c1: bool = True) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.append(fig_a2_raw_fft(out_dir))
    paths.append(fig_a3_envelope(out_dir))
    paths.append(fig_a4_side_by_side(out_dir))
    paths.append(fig_a10_imbalance_dose(out_dir))
    paths.append(fig_b6_leakage_diagnostic(out_dir))
    if run_c1:
        from vibedge.experiments import run_c1_synthetic

        c1 = run_c1_synthetic()
        paths.append(fig_c1_protocol_ladder(out_dir, c1))
        # C2 from last confusion if present
        cms = [r.get("confusion") for r in c1["rows"] if r.get("confusion") and r.get("fold") == "pooled"]
        if cms:
            paths.append(fig_c2_confusion(out_dir, np.asarray(cms[-1]), CLASSES))
        fold_scores = [
            r["accuracy"]
            for r in c1["rows"]
            if r["protocol"] == 3 and r["feature_set"] == "envelope_ratio" and r.get("fold") != "pooled"
        ]
        if fold_scores:
            paths.append(fig_c6_lofo_spread(out_dir, fold_scores))
    # run_c1=False: C1/C2/C6 need C1 results, so they are skipped (no placeholders)
    paths.append(fig_c11_severity(out_dir))
    paths.append(fig_speed_validation(out_dir))
    # deployment degradation with real numbers
    paths.append(_fig_deployment_real(out_dir))
    return paths


def _fig_deployment_real(out_dir: Path) -> Path:
    apply_style()
    from vibedge.deployment_view import to_deployment_view
    from vibedge.experiments import quick_macro_f1

    rows = generate_dataset(n_per_class=4, duration=2.0, base_seed=7)
    pristine_f1 = quick_macro_f1([(x, m) for x, _, m in rows])
    deploy_pairs = []
    for i, (x, _, m) in enumerate(rows):
        y, fs = to_deployment_view(x, m.fs, rng=np.random.default_rng(i))
        deploy_pairs.append((y, SampleMeta_clone(m, fs)))
    deploy_f1 = quick_macro_f1(deploy_pairs)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["pristine", "deployment view"], [pristine_f1, deploy_f1], color=COLORS_3[:2])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1 (logreg, P0 illustrative)")
    ax.set_title("Deployment-view degradation")
    return save_figure(
        fig,
        out_dir,
        "deployment_view_degradation",
        {"pristine_macro_f1": pristine_f1, "deployment_macro_f1": deploy_f1},
    )
