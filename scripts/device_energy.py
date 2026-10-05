#!/usr/bin/env python3
"""Energy per window on the ESP32, from the Uno power-meter capture.

  python3 scripts/device_energy.py                                  # nominal constants
  python3 scripts/device_energy.py --vref 1.083 --vsupply 4.98      # measured with a multimeter
  python3 scripts/device_energy.py calibrate --csv results/real/power_cal.csv --load-ohms 100

Model: low-side shunt R_s in the ESP32's ground return.
  I = (adc / 1024) · V_ref / R_s          (Uno, internal reference)
  P = (V_supply − I · R_s) · I            (power delivered to the ESP32 board)
Blocks are found from the marker: runs ≥ 150 ms are compute blocks; the shorter code pulses
just before each block give its feature set (1 = time_only, 2 = envelope_ratio).
Per block: E = ∫ P dt (trapezoid, vibedge.energy), divided by the windows in the block.
Idle power = median P over marker-low samples at least 500 ms from any marker activity.
Writes results/real/device_energy.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibedge.energy import integrate_energy_j, max_step_error_j  # noqa: E402

CODES = {1: "time_only", 2: "envelope_ratio"}
BLOCK_MIN_S = 0.150
PULSE_MAX_S = 0.100
QUIET_S = 0.5


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, adc, mk = [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            t.append(int(r["t_us"]) * 1e-6)
            adc.append(int(r["adc_x100"]) / 100.0)
            mk.append(int(r["marker"]))
    if not t:
        raise SystemExit(f"{path}: no rows")
    return np.array(t), np.array(adc), np.array(mk)


def runs(mk: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of marker-high runs."""
    d = np.diff(np.concatenate([[0], mk, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def current(adc, vref, rshunt):
    return adc / 1024.0 * vref / rshunt


def cmd_energy(a) -> int:
    t, adc, mk = load(a.csv)
    i = current(adc, a.vref, a.rshunt)
    p = (a.vsupply - i * a.rshunt) * i
    dt_med = float(np.median(np.diff(t)))

    rr = runs(mk)
    blocks, pulses = [], []
    for s, e in rr:
        dur = t[e - 1] - t[s]
        if dur >= BLOCK_MIN_S:
            blocks.append((s, e))
        elif dur <= PULSE_MAX_S:
            pulses.append((s, e))

    # idle: marker low and ≥ QUIET_S from any marker-high sample
    high_t = t[mk == 1]
    if len(high_t):
        idx = np.searchsorted(high_t, t)
        near = np.minimum(np.abs(t - high_t[np.clip(idx - 1, 0, len(high_t) - 1)]),
                          np.abs(t - high_t[np.clip(idx, 0, len(high_t) - 1)]))
        quiet = (mk == 0) & (near >= QUIET_S)
    else:
        quiet = mk == 0
    p_idle = float(np.median(p[quiet])) if quiet.any() else float("nan")

    per_set: dict[str, list[dict]] = {v: [] for v in CODES.values()}
    skipped = []
    prev_end_t = -np.inf
    for s, e in blocks:
        t0, t1 = t[s], t[e - 1]
        code = sum(1 for ps, pe in pulses if prev_end_t < t[ps] < t0 and t0 - t[pe - 1] < 1.5)
        prev_end_t = t1
        name = CODES.get(code)
        dur = t1 - t0
        if name is None or s == 0 or e == len(t):      # undecodable, or cut off by the capture edges
            skipped.append({"start_s": round(float(t0 - t[0]), 3), "duration_s": round(float(dur), 3), "code": code})
            continue
        energy = integrate_energy_j(t[s:e], p[s:e])
        p_mean = energy / dur
        w = a.windows_per_block
        per_set[name].append({
            "start_s": round(float(t0 - t[0]), 3), "duration_s": float(dur), "p_mean_w": p_mean,
            "energy_j": energy, "energy_per_window_mj": energy / w * 1e3,
            "time_per_window_ms": dur / w * 1e3,
            "incremental_energy_per_window_mj": (p_mean - p_idle) * dur / w * 1e3,
        })

    summary = {}
    for name, bl in per_set.items():
        if not bl:
            continue
        def stat(key):
            v = np.array([b[key] for b in bl])
            return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        summary[name] = {
            "n_blocks": len(bl), "windows_per_block": a.windows_per_block,
            "time_per_window_ms": stat("time_per_window_ms"),
            "power_active_w": stat("p_mean_w"),
            "energy_per_window_mj": stat("energy_per_window_mj"),
            "incremental_energy_per_window_mj": stat("incremental_energy_per_window_mj"),
        }
    if {"time_only", "envelope_ratio"} <= summary.keys():
        summary["envelope_over_time_energy_ratio"] = (summary["envelope_ratio"]["energy_per_window_mj"]["mean"]
                                                      / summary["time_only"]["energy_per_window_mj"]["mean"])
    di = float(np.median(np.diff(np.unique(np.round(i, 6))))) if len(np.unique(i)) > 1 else float("nan")
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "csv": str(a.csv.relative_to(ROOT) if a.csv.is_relative_to(ROOT) else a.csv),
        "device": "ESP32, 240 MHz, no radio; 8 stored CWRU windows of 16384 samples (1.37 s at 12 kHz)",
        "constants": {"rshunt_ohm": a.rshunt, "vref_v": a.vref, "vsupply_v": a.vsupply,
                      "vref_source": a.vref_source},
        "sampling": {"median_dt_ms": dt_med * 1e3, "rows": int(len(t)), "duration_s": float(t[-1] - t[0])},
        "power_idle_w": p_idle,
        "per_set": summary,
        "blocks": per_set,
        "skipped_blocks": skipped,
        "accuracy_notes": [
            "Absolute scale depends on V_ref (Uno internal reference, ±10% unless measured) and V_supply; "
            "the envelope/time ratio does not depend on either.",
            f"Trapezoid step error per marker edge ≤ ΔP·dt/2 = {max_step_error_j(1.0, dt_med) * 1e3:.3f} mJ per watt of step.",
            "Whole-board power (regulator, LEDs, USB bridge included); incremental energy subtracts idle.",
        ],
    }
    path = ROOT / "results" / "real" / "device_energy.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"idle {p_idle * 1e3:.1f} mW · blocks used: " + ", ".join(f"{k} {v['n_blocks']}" for k, v in summary.items()
                                                                    if isinstance(v, dict)))
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k:15s} {v['time_per_window_ms']['mean']:7.1f} ms/window  {v['power_active_w']['mean'] * 1e3:6.1f} mW  "
                  f"{v['energy_per_window_mj']['mean']:7.3f} mJ/window (± {v['energy_per_window_mj']['sd']:.3f})")
    if skipped:
        print(f"  skipped {len(skipped)} block(s) (undecodable or cut off by the capture start/end)")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0 if summary else 1


def cmd_calibrate(a) -> int:
    """Known resistor in place of the ESP32 → the effective V_ref that makes the reading right."""
    t, adc, mk = load(a.csv)
    i_meas = current(float(np.median(adc)), a.vref, a.rshunt)
    i_true = a.vsupply / (a.load_ohms + a.rshunt)
    vref = a.vref * i_true / i_meas
    print(f"measured {i_meas * 1e3:.2f} mA, expected {i_true * 1e3:.2f} mA  →  use --vref {vref:.4f}")
    print("(expected current assumes --vsupply is right; measure the Uno 5 V pin if you can)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", choices=["energy", "calibrate"], default="energy")
    p.add_argument("--csv", type=Path, default=ROOT / "results" / "real" / "power_raw.csv")
    p.add_argument("--rshunt", type=float, default=1.0, help="shunt resistance, ohms")
    p.add_argument("--vref", type=float, default=1.1, help="Uno internal reference, volts (measure AREF)")
    p.add_argument("--vsupply", type=float, default=5.0, help="supply at the ESP32 5V pin side, volts")
    p.add_argument("--windows-per-block", type=int, default=80)
    p.add_argument("--load-ohms", type=float, help="calibrate: the known resistor")
    a = p.parse_args()
    a.vref_source = "nominal 1.1 V" if a.vref == 1.1 else "user-supplied"
    if a.mode == "calibrate":
        if not a.load_ohms:
            p.error("calibrate needs --load-ohms")
        return cmd_calibrate(a)
    return cmd_energy(a)


if __name__ == "__main__":
    sys.exit(main())
