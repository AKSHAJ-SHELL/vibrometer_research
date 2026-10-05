#!/usr/bin/env python3
"""On-device timing from the ESP32's serial output, plus an ESTIMATED energy per window.

  1. Flash firmware/vibedge_esp32, open Serial Monitor at 115200, let it run for a few cycles.
  2. Paste everything into results/real/device_serial.txt
  3. python3 scripts/device_timing.py        (or: make device-timing)

Time per window is MEASURED (the ESP32's own microsecond clock, median over cycles of 80 windows).
Energy is ESTIMATED: time × datasheet current × supply voltage (configs/power_estimates.yaml).
Writes results/real/device_timing.json.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "real" / "device_serial.txt"
OUT = ROOT / "results" / "real" / "device_timing.json"

CYCLE = re.compile(r"cycle\s+(\d+)\s+time_only\s+([\d.]+)\s+us/window\s+envelope_ratio\s+([\d.]+)\s+us/window"
                   r"(?:\s+\((\d+) windows per block\))?")
CHECK = re.compile(r"^(time_only|envelope_ratio)\s+in-band worst standardised error\s+(\S+).*predictions\s+(\d+)/(\d+)\s+(PASS|FAIL)")
CHIP = re.compile(r"chip\s+(\S+)\s+rev\s+(\d+),\s+(\d+)\s+cores")
SYSINFO = re.compile(r"CPU\s+(\d+)\s+MHz,\s+free heap\s+(\d+)\s+bytes,\s+PSRAM\s+(\d+)\s+bytes")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else LOG
    if not path.exists():
        print(f"{path} not found — paste the ESP32 Serial Monitor output there (see firmware/README.md)")
        return 2
    text = path.read_text(errors="ignore")
    cycles, checks, sysinfo = [], {}, {}
    for line in text.splitlines():
        line = line.strip()
        if m := CYCLE.search(line):
            cycles.append({"cycle": int(m[1]), "time_only_us": float(m[2]), "envelope_ratio_us": float(m[3]),
                           "windows_per_block": int(m[4]) if m[4] else None})
        elif m := CHECK.match(line):
            checks[m[1]] = {"worst_in_band_std_error": float(m[2]), "predictions": f"{m[3]}/{m[4]}", "result": m[5]}
        elif m := CHIP.search(line):
            sysinfo.update({"chip_model": m[1], "chip_revision": int(m[2]), "cores": int(m[3])})
        elif m := SYSINFO.search(line):
            sysinfo.update({"cpu_mhz": int(m[1]), "free_heap_bytes": int(m[2]), "psram_bytes": int(m[3])})
    device_check = "PASS" if "DEVICE CHECK PASS" in text else ("FAIL" if "DEVICE CHECK FAIL" in text else "not found")
    if not cycles:
        print("no 'cycle …' lines found — let the ESP32 run ~30 s after boot before copying the output")
        return 1
    if device_check != "PASS":
        print(f"WARNING: device parity check is {device_check}; timing is still reported but check the port")

    chip = sysinfo.get("chip_model", "")
    key = "esp32s3" if chip.upper().startswith("ESP32-S3") else ("esp32" if chip.upper().startswith("ESP32") else None)
    if key is None:
        key = sys.argv[2] if len(sys.argv) > 2 else None
    if key is None:
        print("chip model not found in the log (older firmware?) — rerun as: "
              "python3 scripts/device_timing.py <log> esp32   (or esp32s3)")
        return 2
    cfg = yaml.safe_load((ROOT / "configs" / "power_estimates.yaml").read_text())[key]
    v_chip, v_board = cfg["supply_v"], cfg["board_input_v"]
    cur = cfg["current_ma"]

    per_set = {}
    for name in ("time_only", "envelope_ratio"):
        vals = [c[f"{name}_us"] for c in cycles]
        t_ms = statistics.median(vals) / 1000.0
        def mj(ma, v):  # mA × V × ms = µJ → /1000 = mJ
            return ma * v * t_ms / 1000.0
        per_set[name] = {
            "time_per_window_ms": {"median": t_ms, "min": min(vals) / 1000, "max": max(vals) / 1000,
                                   "n_cycles": len(vals)},
            "energy_per_window_mj_estimate": {
                "chip_3v3_point": mj(cur["point"], v_chip),
                "chip_3v3_range": [mj(cur["low"], v_chip), mj(cur["high"], v_chip)],
                "board_5v_lower_bound": mj(cur["point"], v_board),
            },
            "realtime_factor": (16384 / 12000.0) / (t_ms / 1000.0),   # seconds of signal per second of compute
        }
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": f"{chip or key} ({key}), one core at {sysinfo.get('cpu_mhz', 240)} MHz, radio never started; "
                  "8 stored CWRU windows of 16384 samples (1.37 s at 12 kHz)",
        "power_profile": key,
        "device_check": device_check, "parity": checks, "sysinfo": sysinfo,
        "time_source": "MEASURED: esp_timer, median over cycles (80 windows per cycle per feature set)",
        "energy_source": f"ESTIMATED: time × current × voltage; current from {cfg['source']}",
        "power_constants": {"current_ma": cur, "chip_v": v_chip, "board_v": v_board},
        "per_set": per_set,
        "envelope_over_time_ratio": per_set["envelope_ratio"]["time_per_window_ms"]["median"]
                                    / per_set["time_only"]["time_per_window_ms"]["median"],
        "limitations": [
            "Energy is not measured: it assumes the datasheet current for this mode; the real board draws more "
            "(regulator, LED, USB bridge), so the 5 V figure is a lower bound for the dev board.",
            "Both feature sets run the CPU at full load, so the energy ratio equals the time ratio.",
            "Excludes sensor acquisition (windows are stored) and speed estimation (measured speed is passed in).",
        ],
        "cycles": cycles,
    }
    OUT.write_text(json.dumps(out, indent=2))
    for name, v in per_set.items():
        e = v["energy_per_window_mj_estimate"]
        print(f"{name:15s} {v['time_per_window_ms']['median']:8.2f} ms/window (measured, {v['time_per_window_ms']['n_cycles']} cycles)"
              f"   ≈ {e['chip_3v3_point']:.2f} mJ/window chip (est., range {e['chip_3v3_range'][0]:.2f}–{e['chip_3v3_range'][1]:.2f})")
    print(f"envelope_ratio / time_only time = {out['envelope_over_time_ratio']:.2f}×   device check: {device_check}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
