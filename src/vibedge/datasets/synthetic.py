"""Physics-based synthetic vibration generator.

Scoped as a *code-bug detector*: injected BPFO fault must raise BPFO-band
energy; identities BPFO+BPFI = n*fr are asserted in tests. Includes synthetic
tacho so the speed estimator is testable before any download.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from vibedge.features.bearing import GENERIC_6205, BearingGeometry, bearing_freqs
from vibedge.taxonomy import FaultClass, SampleMeta


FaultType = Literal[
    "healthy",
    "imbalance",
    "misalignment",
    "bearing_OR",
    "bearing_IR",
    "bearing_ball",
]


@dataclass
class SyntheticConfig:
    fs: float = 26700.0
    duration: float = 4.0
    fr_hz: float = 30.0  # 1800 RPM
    geo: BearingGeometry = GENERIC_6205
    snr_db: float = 15.0
    resonance_hz: float = 3500.0
    resonance_bw: float = 400.0
    seed: int = 0


def _resonance_impulse_response(fs: float, f0: float, bw: float, n: int) -> np.ndarray:
    t = np.arange(n) / fs
    # damped sinusoid
    tau = 1.0 / (np.pi * bw)
    return np.exp(-t / tau) * np.sin(2 * np.pi * f0 * t)


def _impact_train(
    n: int,
    fs: float,
    rate_hz: float,
    amplitude: float,
    ir: np.ndarray,
    am_freq: float | None = None,
    am_depth: float = 0.5,
    phase: float = 0.0,
) -> np.ndarray:
    """Periodic impacts filtered by structural resonance IR."""
    y = np.zeros(n)
    period = fs / rate_hz
    idx = phase * period
    while idx < n:
        i = int(round(idx))
        amp = amplitude
        if am_freq is not None:
            amp *= 1.0 + am_depth * np.sin(2 * np.pi * am_freq * (i / fs))
        end = min(i + len(ir), n)
        y[i:end] += amp * ir[: end - i]
        idx += period
    return y


def _tacho(n: int, fs: float, fr_hz: float, pulses_per_rev: int = 1) -> np.ndarray:
    """Square-ish tachometer pulse train."""
    t = np.arange(n) / fs
    phase = (t * fr_hz * pulses_per_rev) % 1.0
    return (phase < 0.1).astype(np.float64)


def generate_signal(
    fault: FaultType,
    cfg: SyntheticConfig | None = None,
    severity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, SampleMeta]:
    """Return (acceleration, tacho, meta)."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed + hash(fault) % 10_000)
    n = int(round(cfg.duration * cfg.fs))
    t = np.arange(n) / cfg.fs
    fr = cfg.fr_hz
    ff = bearing_freqs(fr, cfg.geo)

    # base: shaft 1× + small harmonics + noise
    x = 0.3 * np.sin(2 * np.pi * fr * t)
    x += 0.05 * np.sin(2 * np.pi * 2 * fr * t + 0.3)
    x += 0.02 * np.sin(2 * np.pi * 3 * fr * t)

    ir = _resonance_impulse_response(
        cfg.fs, cfg.resonance_hz, cfg.resonance_bw, n=int(0.05 * cfg.fs)
    )

    label = fault
    fault_id = f"synth_{fault}_s{severity:.2f}"

    if fault == "imbalance":
        # dominant 1× ∝ severity (mass)
        x = severity * 1.5 * np.sin(2 * np.pi * fr * t)
        x += 0.05 * np.sin(2 * np.pi * 2 * fr * t)
        label = FaultClass.IMBALANCE.value
    elif fault == "misalignment":
        # elevated 2× radially
        x = 0.4 * np.sin(2 * np.pi * fr * t)
        x += severity * 1.2 * np.sin(2 * np.pi * 2 * fr * t)
        x += 0.3 * np.sin(2 * np.pi * 3 * fr * t)
        label = FaultClass.MISALIGNMENT.value
    elif fault == "bearing_OR":
        x += _impact_train(n, cfg.fs, ff.bpfo, amplitude=severity * 2.0, ir=ir)
        label = FaultClass.BEARING_OR.value
    elif fault == "bearing_IR":
        # amplitude modulated at 1×
        x += _impact_train(
            n,
            cfg.fs,
            ff.bpfi,
            amplitude=severity * 2.0,
            ir=ir,
            am_freq=fr,
            am_depth=0.8,
        )
        label = FaultClass.BEARING_IR.value
    elif fault == "bearing_ball":
        x += _impact_train(
            n,
            cfg.fs,
            ff.ball_fault,
            amplitude=severity * 1.5,
            ir=ir,
            am_freq=ff.ftf,
            am_depth=0.6,
        )
        label = FaultClass.BEARING_BALL.value
    elif fault == "healthy":
        label = FaultClass.HEALTHY.value
    else:
        raise ValueError(fault)

    # noise
    signal_power = np.mean(x**2) + 1e-12
    noise_power = signal_power / (10 ** (cfg.snr_db / 10))
    x = x + rng.normal(0, np.sqrt(noise_power), size=n)

    tacho = _tacho(n, cfg.fs, fr)
    meta = SampleMeta(
        dataset="synthetic",
        file_id=fault_id,
        fault_id=fault_id,
        label=label,
        severity=severity,
        rpm=fr * 60.0,
        fs=cfg.fs,
        geometry={"n": cfg.geo.n, "d": cfg.geo.d, "D": cfg.geo.D, "phi_deg": cfg.geo.phi_deg},
        extras={"fr_hz": fr, "resonance_hz": cfg.resonance_hz},
    )
    return x.astype(np.float64), tacho, meta


def generate_dataset(
    n_per_class: int = 4,
    severities: list[float] | None = None,
    base_seed: int = 0,
    duration: float = 4.0,
    fs: float = 26700.0,
    fr_hz: float = 30.0,
) -> list[tuple[np.ndarray, np.ndarray, SampleMeta]]:
    """Generate a small multi-class synthetic corpus."""
    severities = severities or [0.5, 1.0, 1.5, 2.0]
    faults: list[FaultType] = [
        "healthy",
        "imbalance",
        "misalignment",
        "bearing_OR",
        "bearing_IR",
        "bearing_ball",
    ]
    out: list[tuple[np.ndarray, np.ndarray, SampleMeta]] = []
    k = 0
    for fault in faults:
        for i in range(n_per_class):
            sev = severities[i % len(severities)] if fault != "healthy" else 0.0
            # slight fr jitter per recording for realism
            fr = fr_hz * (1.0 + 0.01 * ((i % 3) - 1))
            cfg = SyntheticConfig(
                fs=fs, duration=duration, fr_hz=fr, seed=base_seed + k
            )
            # unique fault_id per physical "unit"
            x, tacho, meta = generate_signal(fault, cfg, severity=sev)
            meta.fault_id = f"synth_{fault}_unit{i}"
            meta.file_id = f"synth_{fault}_unit{i}_rep{k}"
            meta.severity = sev
            out.append((x, tacho, meta))
            k += 1
    return out


def fr_from_tacho(tacho: np.ndarray, fs: float) -> float:
    """Estimate shaft rate from synthetic / real tacho pulse train."""
    # find rising edges
    d = np.diff((tacho > 0.5).astype(np.int8))
    rises = np.where(d == 1)[0]
    if len(rises) < 2:
        return 0.0
    periods = np.diff(rises) / fs
    return float(1.0 / np.median(periods))
