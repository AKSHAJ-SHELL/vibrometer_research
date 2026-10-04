#!/usr/bin/env python3
"""Write results/real/provenance.json: what produced the current results.

Records git commit (null if this is not a git repo), Python and package
versions, the sha256 of data/sha256sums.txt (a hash over every downloaded
file's hash), per-dataset file counts from that manifest, and a UTC timestamp.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "joblib", "pyyaml", "requests", "h5py"]


def git_commit() -> dict:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return {"commit": out.stdout.strip(), "dirty": bool(dirty.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None, "note": "not a git repository"}


def data_summary() -> dict:
    sums = ROOT / "data" / "sha256sums.txt"
    if not sums.exists():
        return {"sha256sums_txt": None, "note": "data/sha256sums.txt missing"}
    text = sums.read_bytes()
    counts: dict[str, int] = {}
    for line in text.decode().splitlines():
        if "  " in line:
            ds = line.split("  ", 1)[1].split("/", 1)[0]
            counts[ds] = counts.get(ds, 0) + 1
    return {"sha256sums_txt": hashlib.sha256(text).hexdigest(), "files_hashed_by_dataset": counts}


def main() -> int:
    versions = {}
    for pkg in PACKAGES + ["vibedge"]:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "data": data_summary(),
        "command": " ".join(sys.argv[1:]) or "make reproduce",
        "results": sorted(p.name for p in (ROOT / "results" / "real").glob("*.json") if p.name != "provenance.json"),
    }
    path = ROOT / "results" / "real" / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
