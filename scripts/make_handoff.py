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


def device_energy() -> list[str]:
    """On-device section: measured energy if the Uno capture exists, else measured time + estimated energy."""
    eng, tim, exp = RES / "device_energy.json", RES / "device_timing.json", RES / "device_export.json"
    if eng.exists():
        d = json.loads(eng.read_text())
        lines = [f"**Measured** with the Uno power meter. {d['device']}. Constants: shunt {d['constants']['rshunt_ohm']} Ω, "
                 f"V_ref {d['constants']['vref_v']} V ({d['constants']['vref_source']}), supply {d['constants']['vsupply_v']} V. "
                 f"Idle power {d['power_idle_w'] * 1e3:.1f} mW.", "",
                 "| Feature set | Blocks | Time / window | Active power | Energy / window | Incremental energy / window |",
                 "|---|---|---|---|---|---|"]
        for name in ("time_only", "envelope_ratio"):
            v = d["per_set"].get(name)
            if v:
                lines.append(f"| {name} | {v['n_blocks']} | {v['time_per_window_ms']['mean']:.1f} ms | "
                             f"{v['power_active_w']['mean'] * 1e3:.1f} mW | "
                             f"{v['energy_per_window_mj']['mean']:.3f} ± {v['energy_per_window_mj']['sd']:.3f} mJ | "
                             f"{v['incremental_energy_per_window_mj']['mean']:.3f} mJ |")
        return lines
    if tim.exists():
        d = json.loads(tim.read_text())
        c = d["power_constants"]
        lines = [f"{d['device']}. Device parity check: **{d['device_check']}**.", "",
                 f"Time is **measured** ({d['time_source']}). Energy is **estimated**: time × "
                 f"{c['current_ma']['typical']} mA × {c['chip_v']} V (ESP32-S3 datasheet v2.2, Table 5-9; "
                 f"range {c['current_ma']['low']}–{c['current_ma']['high']} mA).", "",
                 "| Feature set | Time / window (measured) | Real-time factor | Energy / window (estimated, chip) | Range | Dev board at 5 V (lower bound) |",
                 "|---|---|---|---|---|---|"]
        for name in ("time_only", "envelope_ratio"):
            v = d["per_set"][name]
            e = v["energy_per_window_mj_estimate"]
            lines.append(f"| {name} | {v['time_per_window_ms']['median']:.2f} ms | {v['realtime_factor']:.0f}× | "
                         f"{e['chip_3v3_typical']:.2f} mJ | {e['chip_3v3_range'][0]:.2f}–{e['chip_3v3_range'][1]:.2f} mJ | "
                         f"≥ {e['board_5v_lower_bound']:.2f} mJ |")
        lines += ["", f"envelope_ratio takes {d['envelope_over_time_ratio']:.2f}× the time (and so, at equal power, the energy) of time_only."]
        return lines
    note = "_Not run yet — see `firmware/README.md`._"
    if exp.exists():
        e = json.loads(exp.read_text())
        hs = e["honest_split_macro_f1_at_this_window_length"]
        note += (f" Exported model: {e['window_samples']}-sample windows ({e['window_s']:.2f} s); "
                 f"honest-split macro-F1 at this length P3 envelope {hs['P3|envelope_ratio']:.3f} vs time {hs['P3|time_only']:.3f}.")
    return [note]


def gateway() -> list[str]:
    """Gateway benchmark rows, one per machine that has run scripts/bench_gateway.py."""
    files = sorted(RES.glob("gateway_*.json"))
    if not files:
        return ["_Not run yet — `make gateway` on the gateway machine._"]
    lines = ["Time is **measured** on each machine; sensors served counts compute only, at a 2 s hop "
             "(4 s windows, 50% overlap). Energy is **estimated** from a published board figure where one exists.", "",
             "| Machine | Feature set | Python, 1 core | Windows / s (all cores) | Sensors in real time | Native C++, 1 core | Est. energy / window |",
             "|---|---|---|---|---|---|---|"]
    for f in files:
        d = json.loads(f.read_text())
        nat = d.get("native_cpp") or {}
        en = d.get("energy_estimate", {}).get("per_set", {})
        for name in ("time_only", "envelope_ratio"):
            v = d["per_set"][name]
            n = nat.get(name) if isinstance(nat, dict) else None
            ntxt = f"{n['us_per_window'] / 1e3:.2f} ms per {n['window_s']:.2f} s" if n else "—"
            etxt = f"{en[name]['board_mj_per_window_at_full_load']:.1f} mJ (board)" if name in en else "—"
            lines.append(f"| {d['machine']} ({d['cores']} cores) | {name} | {v['python_single_core_ms']['median']:.2f} ms | "
                         f"{v['python_throughput_windows_per_s']:,.0f} | ~{v['sensors_realtime']['hop_2s_50pct_overlap']:,.0f} | "
                         f"{ntxt} | {etxt} |")
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
        "## On-device (ESP32-S3)",
        "",
        *device_energy(),
        "",
        "## Gateway (Orange Pi 5 Plus / laptop)",
        "",
        *gateway(),
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
