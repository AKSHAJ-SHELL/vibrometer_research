#!/usr/bin/env python3
"""Build results/handoff/summary.md from the result JSONs.

Every number in the summary is read from results/real/*.json at build time;
nothing numeric is typed here. Limitations are quoted verbatim from
adjustments_and_cliffs.md by their IDs. Missing inputs are reported in the
summary rather than silently skipped.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "real"
OUT = ROOT / "results" / "handoff" / "summary.md"
CLIFFS = ROOT / "adjustments_and_cliffs.md"

HEADLINE = [  # (dataset, protocol, label)
    ("cwru", "3", "CWRU · P3 leave-one-fault-out"),
    ("cwru", "4", "CWRU · P4 leave-one-severity-out"),
    ("paderborn", "3", "Paderborn · P3 leave-one-bearing-out"),
    ("paderborn", "3b", "Paderborn · P3b artificial → real damage"),
]
MODELS = [("logreg", ""), ("GBDT", "_gbdt")]
FIGURES = [  # (file, one line — no numbers)
    ("C1_ladder_real.png", "Pooled macro-F1 across the split ladder (P0 → P4) for each feature set, CWRU and Paderborn."),
    ("speed_ablation.png", "The same honest splits with measured, estimated and rated-speed-prior shaft speed, with 95% fault-level CIs."),
    ("C10_transfer_real.png", "Cross-dataset transfer (train on row, test on column) for time-only vs envelope features."),
    ("deployment_degradation_real.png", "P3 score before and after the simulated deployment sensor (6 kHz LPF, 26.7 kSPS, MEMS noise)."),
    ("A3_envelope_cwru.png", "Real CWRU envelope spectra with BPFO / BPFI / 2×BSF marked — the physics sanity check."),
]
LIMITATIONS = ["C1", "C1b", "C1c", "C6", "C2", "C3"]

missing: list[str] = []


def load(name: str) -> dict | None:
    p = RES / name
    if not p.exists():
        missing.append(f"results/real/{name}")
        return None
    return json.loads(p.read_text())


def pooled(res: dict, protocol: str, fs: str) -> dict | None:
    for r in res["rows"]:
        if r["fold"] == "pooled" and str(r["protocol"]) == protocol and r["feature_set"] == fs:
            return r
    return None


def f_ci(row: dict | None, key: str = "macro_f1", ci_key: str = "ci95") -> str:
    if row is None or row.get(key) is None:
        return "—"
    ci = row.get(ci_key)
    return f"{row[key]:.3f} [{ci['low']:.2f}, {ci['high']:.2f}]" if ci else f"{row[key]:.3f}"


def diff(row: dict | None) -> str:
    d = (row or {}).get("diff_vs_time_only_ci95")
    return f"{d['point']:+.3f} [{d['low']:+.2f}, {d['high']:+.2f}]" if d else "—"


def headline_table() -> list[str]:
    lines = [
        "| Split | Model | envelope_ratio | time_only | Δ (envelope − time) | CWRU excl. healthy: envelope / time |",
        "|---|---|---|---|---|---|",
    ]
    for model, suffix in MODELS:
        for ds, p, label in HEADLINE:
            res = load(f"c1_{ds}{suffix}.json")
            if res is None:
                continue
            env, tim = pooled(res, p, "envelope_ratio"), pooled(res, p, "time_only")
            if env is None:
                continue
            excl = "—"
            if env.get("macro_f1_excl_healthy") is not None:
                excl = (f"{f_ci(env, 'macro_f1_excl_healthy', 'macro_f1_excl_healthy_ci95')} / "
                        f"{f_ci(tim, 'macro_f1_excl_healthy', 'macro_f1_excl_healthy_ci95')}")
            lines.append(f"| {label} | {model} | {f_ci(env)} | {f_ci(tim)} | {diff(env)} | {excl} |")
    return lines


def speed_table() -> list[str]:
    res = load("speed_ablation.json")
    if res is None:
        return ["_speed_ablation.json missing — run `scripts/run_real.py --only speed`._"]
    lines = [
        "| Dataset | Speed source | within 2% | within 5% | locked to 2× | median rel. error | speed features used | Δ envelope − time (first honest split) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ds, modes in res.items():
        for mode, r in modes.items():
            sa = r["speed_accuracy"]
            first = next((row for row in r["rows"] if row["fold"] == "pooled"
                          and row["feature_set"] == "envelope_ratio"), None)
            tag = f"P{first['protocol']}: {diff(first)}" if first else "—"
            lines.append(
                f"| {ds} | {mode} | {sa['within_2pct']:.0%} | {sa['within_5pct']:.0%} | {sa['locked_to_2x']:.0%} | "
                f"{sa['median_rel_error']:.1%} | {sa['used_speed_features']:.0%} | {tag} |"
            )
    return lines


def cost_table() -> list[str]:
    res = load("cost.json")
    if res is None:
        return ["_cost.json missing — run `scripts/run_real.py --only cost`._"]
    lines = [
        f"Measured on one {res['window_s']:g} s window of `{res['window_file']}` at {res['fs']:g} Hz; "
        f"models fit on {res['models_fit_on']}. Host-CPU timing is a relative ranking, not MCU latency.",
        "",
        "| Feature set | Features | Bytes (float32) | Logreg params (model + scaler) | GBDT params (model + scaler) | Extract time / window |",
        "|---|---|---|---|---|---|",
    ]
    for r in res["rows"]:
        lg, gb = r.get("params_logreg", {}), r.get("params_gbdt", {})
        lines.append(
            f"| {r['feature_set']} | {r['n_features']} | {r['feature_bytes_float32']} | "
            f"{lg.get('total', '—')} ({lg.get('model', '—')} + {lg.get('preprocessing', '—')}) | "
            f"{gb.get('total', '—')} ({gb.get('model', '—')} + {gb.get('preprocessing', '—')}) | "
            f"{r['extract_s_per_window'] * 1e3:.2f} ms |"
        )
    return lines


def figures() -> list[str]:
    lines = []
    for name, desc in FIGURES:
        p = ROOT / "figures" / "real" / name
        flag = "" if p.exists() else " **(missing — run `make figures-real`)**"
        lines.append(f"- `figures/real/{name}` — {desc}{flag}")
    return lines


def limitations() -> list[str]:
    if not CLIFFS.exists():
        missing.append("adjustments_and_cliffs.md")
        return ["_adjustments_and_cliffs.md missing._"]
    text = CLIFFS.read_text()
    out = []
    for cid in LIMITATIONS:
        # a cliff is "- **<ID>. ..." up to the next top-level "- **C" bullet or heading
        m = re.search(rf"^- \*\*{re.escape(cid)}\. .*?(?=^- \*\*C\d|^#|\Z)", text, re.S | re.M)
        if m:
            out.append(m.group(0).rstrip())
        else:
            out.append(f"- **{cid}.** _not found in adjustments_and_cliffs.md_")
            missing.append(f"cliff {cid}")
    return out


def main() -> int:
    prov = load("provenance.json")
    body = [
        "# vibedge — results handoff",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `scripts/make_handoff.py` "
        "from `results/real/*.json`. Every number below is read from those files; regenerate with "
        "`make handoff` rather than editing by hand._",
        "",
    ]
    if prov:
        git = prov["git"].get("commit") or prov["git"].get("note", "unknown")
        body += [f"Provenance recorded {prov['timestamp_utc']} (written by `make reproduce` after its run) · git: {git} · Python {prov['python']} · "
                 f"data manifest sha256 `{(prov['data'].get('sha256sums_txt') or 'n/a')[:16]}…`", ""]
    body += [
        "## Headline — pooled macro-F1 with 95% fault-level CI",
        "",
        "Honest splits only (P0 random windows is invalid by design and excluded). Δ is a paired bootstrap "
        "over the same fault resamples. Shaft speed: measured (oracle) — see the next section.",
        "",
        *headline_table(),
        "",
        "## Shaft-speed accuracy per speed source",
        "",
        *speed_table(),
        "",
        "## Cost column (EMC²)",
        "",
        *cost_table(),
        "",
        "## Figures",
        "",
        *figures(),
        "",
        "## Top limitations",
        "",
        "_Quoted verbatim from `adjustments_and_cliffs.md`._",
        "",
        *limitations(),
        "",
        "## AI-use disclosure",
        "",
        "<!-- To be written by the author. -->",
        "",
    ]
    if missing:
        body += ["## Missing inputs", "", *[f"- {m}" for m in missing], ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(body))
    print(f"wrote {OUT.relative_to(ROOT)}" + (f" ({len(missing)} missing inputs)" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
