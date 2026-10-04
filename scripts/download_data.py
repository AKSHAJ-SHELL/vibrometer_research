#!/usr/bin/env python3
"""Download public datasets and record sha256 of every fetched file.

  python scripts/download_data.py --dataset cwru        # 161 .mat files, ~1 GB
  python scripts/download_data.py --dataset mafaulda    # 6 class archives, 12.9 GB compressed
  python scripts/download_data.py --dataset paderborn   # 32 bearing .rar archives, ~5.5 GB
  python scripts/download_data.py --dataset mfpt        # Kaggle mirror (needs ~/.kaggle creds)
  python scripts/download_data.py --dataset all

Downloads resume from a ``.part`` file. Hashes go to data/sha256sums.txt; a file
whose recorded hash no longer matches is reported, never silently replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibedge.datasets.cwru_catalog import CATALOG, CWRU_URL  # noqa: E402

DATA = ROOT / "data"
HASHES_FILE = DATA / "sha256sums.txt"
_hash_lock = threading.Lock()

MAFAULDA_BASE = "http://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/{name}.tgz"
MAFAULDA_PARTS = [
    "normal",
    "imbalance",
    "horizontal-misalignment",
    "vertical-misalignment",
    "underhang",
    "overhang",
]

PADERBORN_BASE = "https://groups.uni-paderborn.de/kat/BearingDataCenter/{code}.rar"
PADERBORN_CODES = (
    [f"K00{i}" for i in range(1, 7)]
    + ["KA01", "KA03", "KA04", "KA05", "KA06", "KA07", "KA08", "KA09", "KA15", "KA16", "KA22", "KA30"]
    + ["KB23", "KB24", "KB27"]
    + ["KI01", "KI03", "KI04", "KI05", "KI07", "KI08", "KI14", "KI16", "KI17", "KI18", "KI21"]
)

MFPT_KAGGLE = "emperorpein/mfpt-fault-datasets"  # third-party mirror; mfpt.org is dead


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_hashes() -> dict[str, str]:
    if not HASHES_FILE.exists():
        return {}
    out = {}
    for ln in HASHES_FILE.read_text().splitlines():
        if "  " in ln:
            digest, name = ln.split("  ", 1)
            out[name] = digest
    return out


def record_hash(name: str, digest: str) -> None:
    with _hash_lock:
        DATA.mkdir(parents=True, exist_ok=True)
        hashes = _read_hashes()
        old = hashes.get(name)
        if old is not None and old != digest:
            print(f"WARNING: sha256 changed for {name}: {old[:12]} -> {digest[:12]}", file=sys.stderr)
        hashes[name] = digest
        HASHES_FILE.write_text("".join(f"{d}  {n}\n" for n, d in sorted(hashes.items())))


def download(url: str, dest: Path, quiet: bool = False) -> Path:
    """Resumable GET into dest (via dest.part). Skips if dest already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    headers = {"User-Agent": "Mozilla/5.0 (vibedge research downloader)"}
    start = part.stat().st_size if part.exists() else 0
    if start:
        headers["Range"] = f"bytes={start}-"
    with requests.get(url, stream=True, timeout=120, headers=headers) as r:
        if r.status_code == 416:  # already complete
            part.rename(dest)
            return dest
        r.raise_for_status()
        mode = "ab" if start and r.status_code == 206 else "wb"
        total = int(r.headers.get("content-length", 0)) + (start if mode == "ab" else 0)
        if not quiet:
            print(f"GET {url} -> {dest.relative_to(ROOT)} ({total / 1e6:.0f} MB)", flush=True)
        with open(part, mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    part.rename(dest)
    return dest


def download_with_retry(url: str, dest: Path, quiet: bool = False, attempts: int = 8) -> Path:
    """Flaky hosts (CWRU, UFRJ) drop connections mid-body; resume from .part."""
    for i in range(attempts):
        try:
            return download(url, dest, quiet=quiet or i > 0)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            print(f"  retry {i + 1}/{attempts} {dest.name}: {type(e).__name__}", flush=True)
            time.sleep(min(2**i, 30))
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code < 500:
                raise
            print(f"  retry {i + 1}/{attempts} {dest.name}: {type(e).__name__}", flush=True)
            time.sleep(min(2**i, 30))
    raise RuntimeError(f"giving up on {url} after {attempts} attempts")


def _fetch_and_hash(url: str, dest: Path, quiet: bool = False) -> Path:
    download_with_retry(url, dest, quiet=quiet)
    record_hash(str(dest.relative_to(DATA)), sha256_file(dest))
    return dest


def cmd_cwru(root: Path) -> None:
    out = root / "cwru"
    jobs = [(CWRU_URL.format(num=num), out / subset / f"{num}.mat") for num, (subset, _name) in CATALOG.items()]
    print(f"CWRU: {len(jobs)} files -> {out.relative_to(ROOT)}/<subset>/")
    with ThreadPoolExecutor(max_workers=3) as ex:
        for i, _ in enumerate(ex.map(lambda j: _fetch_and_hash(*j, quiet=True), jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"  CWRU {i}/{len(jobs)}", flush=True)


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    tool = shutil.which("bsdtar") or shutil.which("tar")
    subprocess.run([tool, "-xf", str(archive), "-C", str(dest)], check=True)


def cmd_mafaulda(root: Path, keep_archives: bool = False) -> None:
    out = root / "mafaulda"
    arch = out / "_archives"
    print("MaFaulDa: licence NOT STATED by UFRJ — cite the authors; do not invent a licence.")
    for name in MAFAULDA_PARTS:
        marker = out / f".extracted_{name}"
        if marker.exists():
            print(f"  {name}: already extracted")
            continue
        a = _fetch_and_hash(MAFAULDA_BASE.format(name=name), arch / f"{name}.tgz")
        print(f"  extracting {name} ...", flush=True)
        _extract(a, out)
        marker.touch()
        if not keep_archives:
            a.unlink()  # hash already recorded
    if not keep_archives and arch.exists() and not any(arch.iterdir()):
        arch.rmdir()


def cmd_paderborn(root: Path, keep_archives: bool = False) -> None:
    out = root / "paderborn"
    arch = out / "_archives"
    print("Paderborn: licence CC BY-NC 4.0")
    for code in PADERBORN_CODES:
        if (out / code).is_dir() and any((out / code).glob("*.mat")):
            continue
        a = _fetch_and_hash(PADERBORN_BASE.format(code=code), arch / f"{code}.rar")
        _extract(a, out)
        if not keep_archives:
            a.unlink()
    if not keep_archives and arch.exists() and not any(arch.iterdir()):
        arch.rmdir()


def cmd_mfpt(root: Path) -> None:
    out = root / "mfpt"
    print(f"MFPT: mfpt.org is dead; using third-party Kaggle mirror {MFPT_KAGGLE}")
    if shutil.which("kaggle") is None:
        raise SystemExit("kaggle CLI not found; pip install kaggle and add ~/.kaggle/kaggle.json")
    subprocess.run(["kaggle", "datasets", "download", "-d", MFPT_KAGGLE, "-p", str(out), "--unzip"], check=True)
    cmd_hash_tree(out, quiet=True)


def cmd_hash_tree(path: Path, quiet: bool = False) -> None:
    path = Path(path)
    for f in sorted(path.rglob("*")):
        if f.is_file() and f.stat().st_size > 0 and not f.name.startswith("."):
            digest = sha256_file(f)
            rel = str(f.relative_to(DATA)) if f.is_relative_to(DATA) else str(f)
            record_hash(rel, digest)
            if not quiet:
                print(f"{digest}  {rel}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["mafaulda", "cwru", "mfpt", "paderborn", "ims", "all"], default="all")
    p.add_argument("--data-root", type=Path, default=DATA)
    p.add_argument("--keep-archives", action="store_true", help="keep .tgz/.rar after extraction")
    p.add_argument("--hash-tree", type=Path, default=None, help="Record sha256 for a tree")
    args = p.parse_args(argv)

    if args.hash_tree:
        cmd_hash_tree(args.hash_tree)
        return 0

    root = args.data_root
    root.mkdir(parents=True, exist_ok=True)
    if args.dataset == "ims":
        print("IMS: https://data.nasa.gov/docs/legacy/IMS.zip (or PHM Society mirror) — not automated")
        return 0
    order = ["cwru", "mfpt", "paderborn", "mafaulda"] if args.dataset == "all" else [args.dataset]
    for ds in order:
        if ds == "cwru":
            cmd_cwru(root)
        elif ds == "mfpt":
            cmd_mfpt(root)
        elif ds == "paderborn":
            cmd_paderborn(root, args.keep_archives)
        elif ds == "mafaulda":
            cmd_mafaulda(root, args.keep_archives)
    return 0


if __name__ == "__main__":
    sys.exit(main())
