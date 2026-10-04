#!/usr/bin/env python3
"""Record the Arduino Uno power meter's serial stream to CSV (run on the Orange Pi or a laptop).

  python3 scripts/power_capture.py --port /dev/ttyACM0 --seconds 180
  python3 scripts/power_capture.py --list          # show serial ports

Writes results/real/power_raw.csv (t_us unwrapped, adc_x100, marker, n). Capture at least
3 full ESP32 cycles: one cycle is ~2 × 3 s idle plus both blocks, so 2–3 minutes is plenty.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import serial
from serial.tools import list_ports

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port")
    p.add_argument("--baud", type=int, default=500000)
    p.add_argument("--seconds", type=float, default=180.0)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "real" / "power_raw.csv")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list or not a.port:
        for port in list_ports.comports():
            print(f"{port.device}\t{port.description}")
        return 0 if a.list else 2

    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows, bad, wraps, last = 0, 0, 0, None
    with serial.Serial(a.port, a.baud, timeout=1) as ser, open(a.out, "w", newline="") as fh:
        time.sleep(2.0)                      # the Uno resets when the port opens
        ser.reset_input_buffer()
        w = csv.writer(fh)
        w.writerow(["t_us", "adc_x100", "marker", "n"])
        end = time.time() + a.seconds
        while time.time() < end:
            line = ser.readline().decode("ascii", "ignore").strip()
            parts = line.split(",")
            if len(parts) != 4 or not all(x.isdigit() for x in parts):
                bad += 1
                continue
            t, adc, mk, n = map(int, parts)
            if last is not None and t < last:
                wraps += 1                    # micros() wrapped at 2^32
            last = t
            w.writerow([t + wraps * 2**32, adc, mk, n])
            rows += 1
            if rows % 10000 == 0:
                print(f"  {rows} rows, {end - time.time():.0f} s left", flush=True)
    print(f"wrote {a.out} ({rows} rows, {bad} unparsed lines skipped)")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
