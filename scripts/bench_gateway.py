#!/usr/bin/env python3
"""Gateway benchmark: how many sensors can one machine (Orange Pi 5 Plus, laptop) serve in real time?

  python3 scripts/bench_gateway.py            (or: make gateway)

Measured on THIS machine, for time_only and envelope_ratio:
  1. Python pipeline (the paper's code): time per 4 s CWRU window on one core (median, 95th percentile).
  2. Throughput with every core busy: windows per second across N worker processes.
  3. Sensors served in real time = throughput × seconds between windows per sensor
     (2 s with 50% overlap as in the paper, 4 s without overlap).
  4. Native C++ (the exact microcontroller code) on one core: time per 1.37 s window; on big.LITTLE
     boards (Orange Pi 5 Plus: Cortex-A55 + A76) once pinned to each core type.
Energy per window is ESTIMATED only on a known board (configs/power_estimates.yaml):
  full-load board power ÷ throughput (board-level, all cores busy), and the increment over idle.

Windows: real CWRU 12 kHz drive-end windows from data/cwru if present, otherwise the bundled
results/real/bench_windows.npz (made on the first run on a machine that has the data).
Writes results/real/gateway_<machine>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibedge.features import FEATURE_SET_GROUPS, extract_features  # noqa: E402
from vibedge.features.bearing import CWRU_6205_DE  # noqa: E402

FS = 12000.0
WINDOW_S = 4.0
BUNDLE = ROOT / "results" / "real" / "bench_windows.npz"
SETS = ("time_only", "envelope_ratio")
N_BUNDLE = 32


def machine_name() -> str:
    model = Path("/proc/device-tree/model")
    if model.exists():
        return model.read_text(errors="ignore").strip("\x00 \n")
    if sys.platform == "darwin":
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        return f"Mac ({brand})" if brand else "Mac"
    return platform.node()


def board_key(name: str) -> str | None:
    n = name.lower().replace(" ", "")
    return "orangepi5plus" if "orangepi5plus" in n else None


def load_windows() -> tuple[np.ndarray, np.ndarray]:
    """(windows float64 [n, 4 s], shaft rate Hz [n]) — real CWRU data, bundled if the dataset is absent."""
    if (ROOT / "data" / "cwru").exists():
        from vibedge.datasets.cwru import load_cwru_directory

        recs = load_cwru_directory(ROOT / "data" / "cwru", target_fs=FS)
        n = int(WINDOW_S * FS)
        picks = recs[:: max(1, len(recs) // N_BUNDLE)][:N_BUNDLE]
        X = np.stack([r.signal[:n] for r in picks])
        fr = np.array([r.meta.rpm / 60.0 for r in picks])
        scale = np.abs(X).max(axis=1, keepdims=True) / 32767.0
        if not BUNDLE.exists():
            BUNDLE.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(BUNDLE, q=np.round(X / scale).astype(np.int16), scale=scale.ravel(), fr=fr,
                                files=np.array([r.meta.file_id for r in picks]))
        return X, fr
    if BUNDLE.exists():
        z = np.load(BUNDLE)
        return z["q"].astype(np.float64) * z["scale"][:, None], z["fr"]
    raise SystemExit("no data/cwru and no results/real/bench_windows.npz — run once where the data is, "
                     "or copy that .npz here")


def _work(args) -> int:
    x, fr, groups = args
    r = extract_features(x, FS, geo=CWRU_6205_DE, fr_hz=fr, groups=groups, slow_seconds=WINDOW_S)
    vec, _ = r.subset(groups)
    return int(np.argmax(vec))          # stand-in for the model: a 39-term dot product is negligible


def single_core(X, fr, groups, passes=3) -> list[float]:
    times = []
    for _ in range(passes):
        for x, f in zip(X, fr):
            t0 = time.perf_counter()
            _work((x, f, groups))
            times.append(time.perf_counter() - t0)
    return times


def throughput(X, fr, groups, workers: int, total: int) -> float:
    tasks = [(X[i % len(X)], fr[i % len(X)], groups) for i in range(total)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_work, tasks[:workers]))                      # warm the workers
        t0 = time.perf_counter()
        list(ex.map(_work, tasks, chunksize=max(1, total // (workers * 8))))
        return total / (time.perf_counter() - t0)


def core_types() -> dict[str, int]:
    """Linux: one representative CPU per core type, keyed by max frequency (big.LITTLE, e.g. RK3588:
    cpu0-3 Cortex-A55, cpu4-7 Cortex-A76). Elsewhere: {} (the OS schedules the run)."""
    base = Path("/sys/devices/system/cpu")
    groups: dict[int, int] = {}
    for cpu in sorted(base.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        f = cpu / "cpufreq" / "cpuinfo_max_freq"
        if f.exists():
            groups.setdefault(int(f.read_text()), int(cpu.name[3:]))
    if len(groups) < 2:
        return {}
    ordered = sorted(groups.items())                       # slowest core type first
    names = ["little", "big", "prime"]
    return {f"{names[i] if i < len(names) else 'core' + str(i)} (cpu{c}, {mhz // 1000} MHz)": c
            for i, (mhz, c) in enumerate(ordered)}


def _parse(stdout: str) -> dict:
    vals = dict(zip(stdout.split()[0::2], stdout.split()[1::2]))
    win_s = float(vals["window_samples"]) / float(vals["fs"])
    return {name: {"us_per_window": float(vals[name]), "window_s": win_s,
                   "realtime_factor": win_s / (float(vals[name]) * 1e-6)} for name in SETS}


def native() -> dict | None:
    """The exact microcontroller C++ code, one core. On big.LITTLE Linux boards, once per core type."""
    cxx = shutil.which("clang++") or shutil.which("g++")
    if not cxx:
        return {"skipped": "no C++ compiler (install g++)"}
    fw = ROOT / "firmware"
    exe = fw / "host_check" / "bench"
    cmd = [cxx, "-O2", "-std=c++17", "-Wno-missing-braces", f"-I{fw / 'vibedge_esp32'}",
           str(fw / "host_check" / "bench.cpp"), str(fw / "vibedge_esp32" / "vibedge_dsp.cpp"), "-o", str(exe)]
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        return {"skipped": "native build failed"}
    out = {"default": _parse(subprocess.run([str(exe)], capture_output=True, text=True).stdout)}
    taskset = shutil.which("taskset")
    for label, cpu in core_types().items():
        if taskset:
            r = subprocess.run([taskset, "-c", str(cpu), str(exe)], capture_output=True, text=True)
            if r.returncode == 0:
                out[label] = _parse(r.stdout)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--total", type=int, default=0, help="windows for the throughput run (default 24 per worker)")
    a = p.parse_args()
    total = a.total or 24 * a.workers

    name = machine_name()
    X, fr = load_windows()
    print(f"{name}: {os.cpu_count()} cores, {len(X)} real CWRU windows of {WINDOW_S:g} s", flush=True)

    per_set = {}
    for s in SETS:
        groups = FEATURE_SET_GROUPS[s]
        t = np.array(single_core(X, fr, groups))
        tp = throughput(X, fr, groups, a.workers, total)
        per_set[s] = {
            "python_single_core_ms": {"median": float(np.median(t) * 1e3), "p95": float(np.percentile(t, 95) * 1e3)},
            "python_throughput_windows_per_s": tp,
            "sensors_realtime": {"hop_2s_50pct_overlap": tp * 2.0, "hop_4s_no_overlap": tp * 4.0},
        }
        print(f"  {s:15s} {per_set[s]['python_single_core_ms']['median']:7.2f} ms/window (1 core)   "
              f"{tp:8.1f} windows/s ({a.workers} workers)  →  ~{tp * 2:,.0f} sensors at a 2 s hop", flush=True)

    nat = native()
    if nat and "skipped" not in nat:
        for label, res in nat.items():
            for st in SETS:
                print(f"  native C++ [{label}] {st:15s} {res[st]['us_per_window']:9.1f} µs per "
                      f"{res[st]['window_s']:.2f} s window (1 core, {res[st]['realtime_factor']:,.0f}× real time)")

    energy = None
    key = board_key(name)
    if key:
        cfg = yaml.safe_load((ROOT / "configs" / "power_estimates.yaml").read_text())[key]
        energy = {"source": cfg["source"], "idle_w": cfg["idle_w"], "full_load_w": cfg["full_load_w"], "per_set": {}}
        for s in SETS:
            tp = per_set[s]["python_throughput_windows_per_s"]
            energy["per_set"][s] = {
                "board_mj_per_window_at_full_load": cfg["full_load_w"] / tp * 1e3,
                "incremental_mj_per_window": (cfg["full_load_w"] - cfg["idle_w"]) / tp * 1e3,
                "watts_per_sensor_at_capacity_2s_hop": cfg["full_load_w"] / (tp * 2.0),
            }
            e = energy["per_set"][s]
            print(f"  est. energy {s:15s} {e['board_mj_per_window_at_full_load']:.2f} mJ/window board-level "
                  f"({e['incremental_mj_per_window']:.2f} above idle)")

    tag = (key or ("mac" if sys.platform == "darwin" else platform.node())).replace(" ", "_").lower()
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": name, "cores": os.cpu_count(), "workers": a.workers, "python": platform.python_version(),
        "windows": {"n": int(len(X)), "window_s": WINDOW_S, "fs": FS,
                    "source": "data/cwru" if (ROOT / "data" / "cwru").exists() else str(BUNDLE.relative_to(ROOT))},
        "time_source": "MEASURED on this machine (time.perf_counter; native: std::chrono)",
        "per_set": per_set, "native_cpp": nat,
        "energy_estimate": energy or {"skipped": "no published power figure for this machine in configs/power_estimates.yaml"},
        "notes": ["Sensors served assume compute is the only limit (network, I/O and OS overhead ignored).",
                  "Python feature extraction only; the model is a dot product of ≤ 40 terms (negligible)."],
    }
    path = ROOT / "results" / "real" / f"gateway_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
