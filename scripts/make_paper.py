#!/usr/bin/env python3
"""Build the IEEE paper (paper/main.tex → paper/main.pdf).

1. Reads results/real/*.json and writes paper/generated/numbers.tex (one \\val{key} per number used in the
   text) and paper/generated/tab_*.tex (every table). No number in the paper is typed by hand.
2. Renders the figures at 300 dpi into paper/figures/ (titles without internal codes).
3. Compiles with tectonic (brew install tectonic).

  python3 scripts/make_paper.py            (or: make paper)
  python3 scripts/make_paper.py --no-figs  (skip re-rendering figures)
A number whose result file is missing prints as a bold ??key in the PDF and is listed at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "real"
PAPER = ROOT / "paper"
GEN = PAPER / "generated"

vals: dict[str, str] = {}
missing: list[str] = []


def load(name: str) -> dict | None:
    p = RES / name
    if not p.exists():
        missing.append(name)
        return None
    return json.loads(p.read_text())


def f3(x) -> str:
    return f"{x:.3f}"


def f2(x) -> str:
    return f"{x:.2f}"


def ci(c) -> str:
    return f"[{c['low']:.2f}, {c['high']:.2f}]"


def sgn(x, nd=2) -> str:
    s = f"{x:+.{nd}f}"
    return s.replace("-", "$-$")


def dci(d) -> str:
    return f"{sgn(d['point'])} [{sgn(d['low'])}, {sgn(d['high'])}]"


def pooled(res, protocol, fs):
    for r in res["rows"]:
        if r["fold"] == "pooled" and str(r["protocol"]) == str(protocol) and r["feature_set"] == fs:
            return r
    return None


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


# ------------------------------------------------------------------ numbers and tables
def build_numbers_and_tables() -> None:
    global vals
    GEN.mkdir(parents=True, exist_ok=True)
    tabs: dict[str, str] = {}

    c1 = {(ds, m): load(f"c1_{ds}{'' if m == 'logreg' else '_gbdt'}.json")
          for ds in ("cwru", "paderborn") for m in ("logreg", "gbdt")}

    # --- data
    cw = c1[("cwru", "logreg")]
    pb = c1[("paderborn", "logreg")]
    ma = load("c1_mafaulda.json")
    if cw:
        vals |= {"cwru-recs": str(cw["n_recordings"]), "cwru-wins": str(cw["n_windows"]), "cwru-ids": str(cw["n_fault_identities"])}
    if pb:
        vals |= {"pb-recs": f"{pb['n_recordings']:,}", "pb-ids": str(pb["n_fault_identities"])}
    if ma:
        vals |= {"ma-recs": f"{ma['n_recordings']:,}"}

    # --- headline numbers for the text (logreg unless noted)
    for (ds, m), res in c1.items():
        if not res:
            continue
        tag = f"{ds}-{m}"
        protos = sorted({str(r["protocol"]) for r in res["rows"] if r["fold"] == "pooled"})
        for p in protos:
            for fs, short in (("envelope_ratio", "env"), ("time_only", "time"), ("full", "full")):
                r = pooled(res, p, fs)
                if r is None:
                    continue
                vals[f"{tag}-P{p}-{short}-f1"] = f3(r["macro_f1"])
                if "ci95" in r:
                    vals[f"{tag}-P{p}-{short}-ci"] = ci(r["ci95"])
                if "diff_vs_time_only_ci95" in r:
                    vals[f"{tag}-P{p}-diff"] = dci(r["diff_vs_time_only_ci95"])
                if r.get("macro_f1_excl_healthy") is not None:
                    vals[f"{tag}-P{p}-{short}-xh"] = f3(r["macro_f1_excl_healthy"])
                    vals[f"{tag}-P{p}-{short}-xhci"] = ci(r["macro_f1_excl_healthy_ci95"])
                if "per_class_f1" in r:
                    for cls, v in r["per_class_f1"].items():
                        vals[f"{tag}-P{p}-{short}-pc-{cls}"] = f2(v)
                vals[f"{tag}-P{p}-base"] = f3(r["majority_baseline_f1"])
                vals[f"{tag}-P{p}-acc-{short}"] = f3(r["accuracy"])

    # Table: headline with CIs
    rows = []
    for m, mname in (("logreg", "LR"), ("gbdt", "GBDT")):
        for ds, p, label in (("cwru", "3", "CWRU, P3"), ("cwru", "4", "CWRU, P4"),
                             ("paderborn", "3", "Paderborn, P3"), ("paderborn", "3b", "Paderborn, P3b")):
            res = c1[(ds, m)]
            if not res:
                continue
            e, t = pooled(res, p, "envelope_ratio"), pooled(res, p, "time_only")
            sig = r"$^{*}$" if e["diff_vs_time_only_ci95"]["low"] > 0 else ""
            rows.append(f"{label} & {mname} & {f3(e['macro_f1'])} {ci(e['ci95'])} & {f3(t['macro_f1'])} {ci(t['ci95'])} & "
                        f"{dci(e['diff_vs_time_only_ci95'])}{sig} \\\\")
    tabs["headline"] = "\n".join([
        r"\begin{table*}[t]", r"\caption{Pooled macro-F1 on the honest splits with 95\% fault-level bootstrap intervals, measured shaft speed. "
        r"$\Delta$ is envelope\_ratio minus time\_only from a paired bootstrap; $^{*}$ marks intervals that exclude zero.}",
        r"\label{tab:headline}", r"\centering", r"\begin{tabular}{llccc}", r"\hline",
        r"Split & Model & envelope\_ratio & time\_only & $\Delta$ (95\% CI) \\", r"\hline", *rows, r"\hline",
        r"\end{tabular}", r"\end{table*}"])

    # Table: ladder (logreg, both datasets)
    if cw and pb:
        names = {"0": "P0 random windows", "1": "P1 leave-one-recording-out", "2": "P2 leave-one-load-out",
                 "3": "P3 leave-one-fault-out", "3b": "P3b artificial $\\rightarrow$ real", "4": "P4 leave-one-severity-out"}
        lrows = []
        for p in ("0", "1", "2", "3", "3b", "4"):
            cells = []
            for res in (cw, pb):
                e, t = pooled(res, p, "envelope_ratio"), pooled(res, p, "time_only")
                cells.append(f"{f3(e['macro_f1'])} & {f3(t['macro_f1'])}" if e else "-- & --")
            lrows.append(f"{names[p]} & {cells[0]} & {cells[1]} \\\\")
        tabs["ladder"] = "\n".join([
            r"\begin{table}[t]", r"\caption{Pooled macro-F1 along the split ladder (logistic regression, measured shaft speed). "
            r"Env.\ = envelope\_ratio, Time = time\_only; -- = protocol not applicable to the dataset.}",
            r"\label{tab:ladder}", r"\centering", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcccc}", r"\hline",
            r" & \multicolumn{2}{c}{CWRU} & \multicolumn{2}{c}{Paderborn} \\",
            r"Protocol & Env. & Time & Env. & Time \\", r"\hline", *lrows, r"\hline", r"\end{tabular}", r"\end{table}"])

    # Table: data
    tr0 = load("transfer.json")
    mf = (tr0 or {}).get("envelope_ratio", {}).get("counts", {}).get("mfpt")
    if cw and pb and ma and mf:
        tabs["data"] = "\n".join([
            r"\begin{table}[t]", r"\caption{Datasets. A fault identity is one physical damaged bearing (or one rig configuration).}",
            r"\label{tab:data}", r"\centering", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrl}", r"\hline",
            r"Dataset & Recordings & Fault ids & Role \\", r"\hline",
            f"CWRU 12 kHz drive end & {cw['n_recordings']} & {cw['n_fault_identities']} & main results \\\\",
            f"Paderborn & {pb['n_recordings']:,} & {pb['n_fault_identities']} & main results \\\\",
            f"MaFaulDa & {ma['n_recordings']:,} & {ma['n_fault_identities']} & held-out imbalance severity \\\\",
            f"MFPT & {mf['recordings']} & {mf['fault_ids']} & cross-dataset transfer \\\\",
            r"\hline", r"\end{tabular}", r"\end{table}"])

    # --- speed ablation
    sp = load("speed_ablation.json")
    if sp:
        srows = []
        label = {"oracle": "measured", "estimated": "estimated, 10--60 Hz", "prior": "estimated, rated $\\pm$10\\%"}
        for ds, dname in (("cwru", "CWRU"), ("paderborn", "Paderborn")):
            for mode in ("oracle", "prior", "estimated"):
                r = sp[ds][mode]
                sa = r["speed_accuracy"]
                deltas = []
                for row in r["rows"]:
                    if row["fold"] == "pooled" and row["feature_set"] == "envelope_ratio":
                        deltas.append(f"P{row['protocol']} {dci(row['diff_vs_time_only_ci95'])}")
                        vals[f"speed-{ds}-{mode}-P{row['protocol']}-diff"] = dci(row["diff_vs_time_only_ci95"])
                        vals[f"speed-{ds}-{mode}-P{row['protocol']}-env-f1"] = f3(row["macro_f1"])
                vals[f"speed-{ds}-{mode}-w2"] = f"{sa['within_2pct'] * 100:.0f}\\%"
                vals[f"speed-{ds}-{mode}-2x"] = f"{sa['locked_to_2x'] * 100:.0f}\\%"
                vals[f"speed-{ds}-{mode}-used"] = f"{sa['used_speed_features'] * 100:.0f}\\%"
                srows.append(f"{dname} & {label[mode]} & {sa['within_2pct'] * 100:.0f}\\% & {sa['locked_to_2x'] * 100:.0f}\\% & "
                             f"{sa['used_speed_features'] * 100:.0f}\\% & {'; '.join(deltas)} \\\\")
        tabs["speed"] = "\n".join([
            r"\begin{table*}[t]", r"\caption{Shaft-speed source versus envelope advantage (logistic regression). ``Within 2\%'' is the share "
            r"of windows whose speed is within 2\% of the measured value; ``used'' is the share that keeps the speed-dependent features "
            r"(the rest fall back to speed-free features); $\Delta$ as in Table~\ref{tab:headline}.}",
            r"\label{tab:speed}", r"\centering", r"\begin{tabular}{llcccl}", r"\hline",
            r"Dataset & Speed source & Within 2\% & Locked to 2$\times$ & Used & $\Delta$ envelope $-$ time (95\% CI) \\",
            r"\hline", *srows, r"\hline", r"\end{tabular}", r"\end{table*}"])

    # --- cost
    cost = load("cost.json")
    if cost:
        crow = []
        for r in cost["rows"]:
            lg, gb = r["params_logreg"], r["params_gbdt"]
            crow.append(f"{tex_escape(r['feature_set'])} & {r['n_features']} & {r['feature_bytes_float32']} & "
                        f"{lg['total']} & {gb['total']:,} & {r['extract_s_per_window'] * 1e3:.2f} \\\\")
            vals[f"cost-{r['feature_set']}-params"] = str(lg["total"])
            vals[f"cost-{r['feature_set']}-bytes"] = str(r["feature_bytes_float32"])
            vals[f"cost-{r['feature_set']}-nf"] = str(r["n_features"])
        tabs["cost"] = "\n".join([
            r"\begin{table}[t]", r"\caption{Cost per feature set: features, feature-vector bytes (float32), parameters "
            r"(model plus standardisation) and host extraction time per 4 s window (laptop CPU, relative only).}",
            r"\label{tab:cost}", r"\centering", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrr}", r"\hline",
            r"Set & Feat. & Bytes & LR par. & GBDT par. & ms \\", r"\hline", *crow, r"\hline", r"\end{tabular}", r"\end{table}"])

    # --- deployment view
    dep = load("deployment.json")
    if dep:
        for ds in ("cwru", "paderborn"):
            for k in ("pristine", "deployment"):
                for fs, short in (("envelope_ratio", "env"), ("time_only", "time")):
                    r = pooled(dep[ds][k], 3, fs)
                    vals[f"dep-{ds}-{k}-{short}"] = f3(r["macro_f1"])

    # --- transfer
    tr = load("transfer.json")
    if tr:
        for fs, short in (("envelope_ratio", "env"), ("time_only", "time")):
            M, names = tr[fs]["scores"], tr[fs]["datasets"]
            for i, a in enumerate(names):
                for j, b in enumerate(names):
                    vals[f"tr-{short}-{a}-{b}"] = f2(M[i][j])
            off = [M[i][j] for i in range(len(names)) for j in range(len(names)) if i != j]
            vals[f"tr-{short}-offmax"] = f2(max(off))

    # --- MaFaulDa severity
    sev = load("severity_mafaulda.json")
    if sev:
        for d, v in sev["designs"].items():
            for fs, short in (("envelope_ratio", "env"), ("time_only", "time"), ("full", "full")):
                vals[f"sev-{d}-{short}"] = f3(v["by_set"][fs]["macro_f1"])
            vals[f"sev-{d}-base"] = f3(v["by_set"]["full"]["majority_baseline_f1"])

    # --- loss curves
    for ds in ("cwru", "paderborn"):
        lc = load(f"loss_curves_{ds}.json")
        if lc:
            for fs, short in (("envelope_ratio", "env"), ("time_only", "time")):
                c = lc["curves"][f"{fs}|P3"]
                vals[f"lc-{ds}-{short}-best"] = str(c["best_stage_by_heldout_loss"])
                vals[f"lc-{ds}-{short}-final"] = f2(c["heldout_logloss"][-1])
                vals[f"lc-{ds}-{short}-min"] = f2(min(x for x in c["heldout_logloss"] if x == x))
                p0 = lc["curves"][f"{fs}|P0"]
                vals[f"lc-{ds}-{short}-p0f1"] = f2(p0["heldout_macro_f1"][-1])
            vals[f"lc-{ds}-nest"] = str(lc["n_estimators"])

    # --- seed stability
    ss = load("seed_stability.json")
    if ss:
        sds = [v["sd"] for d in ss["results"].values() for v in d.values()]
        vals["seed-maxsd"] = f"{max(sds):.3f}"
        vals["seed-n"] = str(len(ss["seeds"]))

    # --- gateway (Orange Pi 5 Plus)
    gw = load("gateway_orangepi5plus.json")
    if gw:
        g = []
        for fs in ("time_only", "envelope_ratio"):
            v = gw["per_set"][fs]
            short = "env" if fs == "envelope_ratio" else "time"
            vals[f"gw-{short}-py1"] = f2(v["python_single_core_ms"]["median"])
            vals[f"gw-{short}-tp"] = f"{v['python_throughput_windows_per_s']:,.0f}"
            vals[f"gw-{short}-sensors"] = f"{v['sensors_realtime']['hop_2s_50pct_overlap']:,.0f}"
            nat = gw.get("native_cpp") or {}
            cores = {k.split(" (")[0]: res[fs] for k, res in nat.items() if isinstance(res, dict) and fs in res}
            for core, r in cores.items():
                vals[f"gw-{short}-cpp-{core}"] = f2(r["us_per_window"] / 1e3)
                vals[f"gw-{short}-cpp-{core}-rt"] = f"{r['realtime_factor']:,.0f}"
            en = gw.get("energy_estimate", {}).get("per_set", {}).get(fs)
            if en:
                vals[f"gw-{short}-mj"] = f2(en["board_mj_per_window_at_full_load"])
                vals[f"gw-{short}-mj-inc"] = f2(en["incremental_mj_per_window"])
                vals[f"gw-{short}-wps"] = f"{en['watts_per_sensor_at_capacity_2s_hop'] * 1e3:.2f}"
            little = cores.get("little", {}).get("us_per_window")
            big = cores.get("big", {}).get("us_per_window")
            g.append(f"{tex_escape(fs)} & {v['python_single_core_ms']['median']:.2f} & "
                     f"{v['python_throughput_windows_per_s']:,.0f} & {v['sensors_realtime']['hop_2s_50pct_overlap']:,.0f} & "
                     f"{little / 1e3 if little else float('nan'):.2f} & {big / 1e3 if big else float('nan'):.2f} & "
                     f"{en['board_mj_per_window_at_full_load'] if en else float('nan'):.1f} \\\\")
        vals["gw-machine"] = tex_escape(gw["machine"])
        vals["gw-cores"] = str(gw["cores"])
        tabs["gateway"] = "\n".join([
            r"\begin{table*}[t]", r"\caption{Orange Pi 5 Plus edge gateway. Python: the full pipeline per 4 s window on one core, and windows "
            r"per second with every core busy; sensors served in real time at a 2 s hop (compute only). C++: the ported pipeline per "
            r"1.37 s window pinned to one Cortex-A55 (little) or Cortex-A76 (big) core. Energy: estimated, full-load board power "
            r"divided by throughput.}", r"\label{tab:gateway}", r"\centering", r"\begin{tabular}{lrrrrrr}", r"\hline",
            r"Feature set & Python ms & Windows/s & Sensors & C++ A55 ms & C++ A76 ms & mJ/window (est.) \\",
            r"\hline", *g, r"\hline", r"\end{tabular}", r"\end{table*}"])
    else:
        tabs["gateway"] = r"% gateway table: results/real/gateway_orangepi5plus.json not present yet"

    # write
    lines = ["% Generated by scripts/make_paper.py from results/real/*.json — do not edit.",
             r"\makeatletter",
             r"\newcommand{\val}[1]{\ifcsname vb@#1\endcsname\csname vb@#1\endcsname\else\textbf{??#1}\fi}",
             r"\newif\ifgateway" + ("\n\\gatewaytrue" if (RES / "gateway_orangepi5plus.json").exists() else "")]
    for k, v in sorted(vals.items()):
        lines.append(f"\\expandafter\\def\\csname vb@{k}\\endcsname{{{v}}}")
    lines.append(r"\makeatother")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    for name, body in tabs.items():
        (GEN / f"tab_{name}.tex").write_text(body + "\n")
    print(f"numbers: {len(vals)} values, tables: {', '.join(sorted(tabs))}")


def render_figures() -> None:
    env = dict(os.environ, VIBEDGE_PAPER="1", VIBEDGE_FIG_DPI="300", VIBEDGE_FIG_OUT=str(PAPER / "figures"))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_real_figures.py")], env=env, check=True,
                   capture_output=True)
    print("figures: rendered at 300 dpi into paper/figures/")


def compile_pdf() -> bool:
    tect = shutil.which("tectonic")
    if not tect:
        print("tectonic not found (brew install tectonic) — LaTeX sources are ready in paper/")
        return False
    r = subprocess.run([tect, "--keep-logs", "main.tex"], cwd=PAPER, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        return False
    print("pdf: paper/main.pdf")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    build_numbers_and_tables()
    if not a.no_figs:
        render_figures()
    ok = True if a.no_pdf else compile_pdf()
    if missing:
        print("missing result files (their numbers print as ??):", ", ".join(missing))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
