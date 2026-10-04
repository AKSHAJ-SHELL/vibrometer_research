# Adjustments and cliffs

This is a record of what was changed in `vibedge`, and why, on 2026-09-29.
It also lists the caveats ("cliffs") that still limit what the results can claim.

Rerun everything with `make real && make figures-real`.
Results go to `results/real/`, and figures to `figures/real/`.

---

## 1. Evaluation fixes

These change the paper's numbers.

| # | Problem | Fix | Where |
|---|---|---|---|
| E1 | The majority baseline was scored on the **test** labels. It gave `1.0` on every single-class fold of `c1_results.json`. | The baseline is fit on each fold's train labels (ties → lowest label). Without `y_train`, `majority_baseline_metrics` warns and scores in-sample. `evaluate_predictions` leaves the baseline `None` unless `y_train` is given. | `evaluate.py` |
| E2 | Leave-one-group-out results were averaged per fold. With one-sample folds that equals accuracy. | The headline is **pooled**: all folds' predictions are concatenated and scored once. It is the row with `fold: "pooled"` in every results file. | `evaluate.pooled_evaluation`, `experiments.score_protocol` |
| E3 | Per-fold macro-F1 was reported for folds where it is undefined. | Per-fold `macro_f1` is `None` when a fold has < 2 test samples or < 2 classes. Per-fold `majority_baseline_f1` stays numeric, because the old `tests/test_c1_experiment.py` required it. | `evaluate.fold_macro_f1_is_defined` |
| E4 | Macro-F1 averaged over all 6 taxonomy classes. Absent classes counted as F1 = 0, which shrinks CWRU (4 classes) by a factor of 4/6. | Macro-F1 averages over the classes present in the dataset. | `experiments`, `real_experiments` |
| E6 | `bootstrap_ci_fault_level` used `np.isin`, so a fault drawn twice counted once. That makes the intervals too narrow. | New `bootstrap_indices` keeps duplicates. Macro-F1 in each draw averages over the classes present in that draw. New `paired_bootstrap_diff` computes the CI of envelope_ratio − time_only with the same fault resample for both. Every pooled honest-split row gets `ci95`, and the envelope row gets `diff_vs_time_only_ci95`. | `evaluate.py`, `real_experiments._score_all_sets` |
| E5 | The train-majority baseline under leave-one-out on balanced classes is always wrong (pooled F1 = 0). This is correct behaviour, but it looks like a bug. | It is documented in the results `notes` and in DESIGN.md. Say it in the methods section. | — |

## 2. Leakage fixes

| # | Problem | Fix | Where |
|---|---|---|---|
| L1 | Nothing checked that a recording is never on both sides of a split. | `make_splits(protocol, ...)` builds every fold of protocols 0–5 from window metadata. For protocols ≥ 1 it raises `LeakageError` naming the offending recording ids. Protocol 3 is also checked by fault identity. | `splits.py`, `windowing.py` |
| L2 | Protocol 4 dealt healthy data to train/test **per window**, which splits one recording across both sides. | Healthy recordings are dealt whole, round-robin. The synthetic frozen manifests are unchanged; they still verify with `--require-frozen`. | `splits.protocol4_held_out_severity` |
| L3 | CWRU fault identity ignored outer-race position. | `@3`, `@6` and `@12` are separate physical faults (`cwru_OR_0.007_@6`). Identity ignores load **and** sampling rate: 12k and 48k use the same seeded bearings, so over-grouping is the conservative direction. | `datasets/cwru.py` |
| L4 | MaFaulDa gave every underhang and overhang bearing fault the same identity. | Identity is one per physical bearing (`mafaulda_underhang_outer_race`). The `0g/6g/20g/35g` subfolders are imbalance mass **added** to the same bearing, not separate faults. | `datasets/mafaulda.py` |
| L5 | MFPT identity was per file, so the same bearing at different loads counted as different faults. | Identity is per fault type (conservative). This makes protocol 3 infeasible on MFPT, and `PROTOCOL_FEASIBILITY` now says so. | `datasets/mfpt.py`, `taxonomy.py` |

## 3. Loader and data fixes

| # | Problem | Fix |
|---|---|---|
| D1 | Downloaded CWRU files are **numbered** (`105.mat`). Filename parsing would have labelled every one of them healthy. | `datasets/cwru_catalog.py` maps number → (subset, fault). It was scraped from the four CWRU data pages (161 files). |
| D2 | CWRU assumed 48 kHz for every file. The 12k sets are 12 kHz. | The rate comes from the subset. `target_fs` resamples, so the 48 kHz normal baseline can be mixed with 12k faults without sampling rate becoming a class cue. |
| D3 | `99.mat` also contains `X098_*` arrays, and the loader read file 98's data. | The loader reads only the array named after the file number. |
| D4 | Several CWRU files contain no array named after their number. | Added to `KNOWN_BAD` after inspecting the download: `174.mat` (only `X173_*`) and `3002–3008.mat` (same defect as `3001.mat`). |
| D5 | Paderborn `KB*` was mapped to **ball**. `KB` means *combined* inner + outer damage, and Paderborn has no rolling-element faults. | `KB*` raises in `map_paderborn` and is excluded (3 bearings). |
| D6 | The Paderborn loader returned **zero** records: data sits under `mat[<stem>].Y`, not a top-level `Y`. | Reads `vibration_1` from `mat[stem].Y`. Speed comes from the filename (`N09` → 900 rpm). Artificial/real damage uses the published lists, not a prefix heuristic. |
| D7 | Paderborn geometry was a generic 9-ball bearing. | The 6203 bearing: n = 8, d = 6.75 mm, D = 28.55 mm. |
| D8 | MFPT used 9-ball geometry. Any unrecognised folder defaulted to "healthy". `5 - Analyses` duplicates a file. | Geometry is n = 8, d = 0.235 in, D = 1.245 in. Shaft rate and load come from each file. Only the labelled rig folders are loaded. |
| D9 | MaFaulDa used 9-ball geometry. | The MFS rig bearing: n = 8, d = 0.7145 cm, D = 2.8519 cm. This reproduces the published BPFO = 2.998 × fr. |
| D10 | MaFaulDa shaft rate was unused. | Taken from the filename (`12.288.csv` = 12.288 Hz). |
| D11 | The downloader was manual-only. | `scripts/download_data.py --dataset {cwru,mfpt,paderborn,mafaulda,all}` resumes interrupted downloads and retries on dropped connections and 5xx errors. It records sha256 in `data/sha256sums.txt` and warns if a hash changes. |

## 4. Feature fixes

| # | Problem | Fix |
|---|---|---|
| F1 | Kurtosis was Fisher with bias correction: −1.500015 for a unit sine. | Kurtosis is **Pearson** with population moments (Gaussian = 3, sine = 1.5). `KURTOSIS_CONVENTION = "pearson"`. Skewness also uses population moments. State this in the paper. |
| F2 | There was no unwindowed amplitude spectrum or Parseval check. | Added `single_sided_amplitude` (2·\|X\|/N, DC and Nyquist not doubled) and `spectral_energy`. |
| F3 | `extract_features` always computed every group, so cost timing was meaningless. | A `groups=` argument computes only the requested groups. `FEATURE_SET_GROUPS` now lives in `vibedge.features`. |

## 5. New pieces

- **`cost.py`**: per feature set, the feature count, float32 bytes, model parameter count (logreg, GBDT; scaler counted separately) and host extraction time per window.
- **`energy.py`**: trapezoid energy for INA219 power traces, plus a bound on the step error.
- **`windowing.py`**: cuts recordings into windows that remember their source recording.
- **`real_experiments.py`, `scripts/run_real.py`**: C1 on CWRU and Paderborn, MaFaulDa held-out severity, cross-dataset transfer, and the deployment view. Features are extracted once per dataset and cached in `results/cache/`. Splits are frozen to `configs/frozen/<dataset>/`.
- **`figures_real.py`, `scripts/make_real_figures.py`**: the figures in `figures/real/`.
- **GBDT (tier 2)**: `run_real.py --only gbdt` writes `c1_{cwru,paderborn}_gbdt.json` and the figure `C1_ladder_real_gbdt.png`. It uses 100 trees, depth 4, learning rate 0.1.
- **CWRU healthy-excluded score**: P3/P4 pooled rows in `c1_cwru.json` and `c1_cwru_gbdt.json` also carry `macro_f1_excl_healthy` (IR/OR/ball) with its own fault-level `ci95`. The reason is in DESIGN.md.
- **`make reproduce`**: re-extracts every feature table (`--refresh`), reruns everything, redraws, and writes `results/real/provenance.json`: git (null here, since this is not a git repo), Python and package versions, the sha256 of `data/sha256sums.txt`, and a timestamp.
- **`make handoff`**: `scripts/make_handoff.py` builds `results/handoff/summary.md`. Every number is read from `results/real/*.json`, and limitations are quoted from this file by ID.
- **`cost.json`**: `run_real.py --only cost` writes the real-data cost column (one CWRU window, models fit on CWRU).
- **Housekeeping (2026-09-30)**:
  - Moved 74 duplicate hash-named CWRU manifests to `configs/frozen/_superseded/cwru/` (see its README). The top-level synthetic manifests are live and stay put.
  - Deleted the stale `results/c1_cwru.json` and `results/cost_table_cwru.json`.
  - Removed every random-number or made-up placeholder from `figures.py` (C6, C10, C2), and deleted their PNGs.
  - `make_figures.py --synthetic` is no longer forced on.
- **Loss curves**: `run_real.py --only curves` writes `loss_curves_{cwru,paderborn}.json` and the figure `loss_curves_gbdt.png`. GBDT runs 300 stages, and after each stage it records the training log-loss (mean over folds), plus the held-out log-loss and macro-F1 (pooled over folds), for P0 and P3.

---

## Cliffs — caveats that limit the claims

### Evaluation design

- **C1. Speed source (re-measured 2026-10-03 with the rewritten estimator).** `results/real/speed_ablation.json` and `figures/real/speed_ablation.png` compare three speed sources (logistic regression, 95% fault-level CI):
  - **measured** (oracle): Paderborn P3 Δ(envelope − time) = **+0.24 [+0.15, +0.33]**; CWRU P4 Δ = **+0.44 [+0.28, +0.61]**; CWRU P3 Δ = +0.16 [−0.06, +0.40].
  - **CWRU, estimated with the rated-speed prior** (±10%): 100% of windows within 2% of the true speed, and the results equal the measured-speed ones (P4 Δ +0.45 [+0.29, +0.62]). **On CWRU the claim holds without a tachometer.**
  - **CWRU, estimated over the generic 10–60 Hz range**: only 12% within 2%, and 67% lock to 2×. The likely cause is the 60 Hz mains line at ≈ 2 × 29.95 Hz. The envelope advantage still largely survives (P4 Δ +0.39 [+0.22, +0.68]), because features at 2× the fault frequency overlap the true fault's second harmonic.
  - **Paderborn, estimated**: the shaft line is weak, so confidence is usually low (median 0) and the speed-free fallback runs on about 60% of windows. The envelope advantage mostly disappears: P3 Δ +0.05 [−0.02, +0.12] generic, +0.04 [−0.03, +0.11] with the prior; P3b with the prior −0.03 [−0.09, +0.02]. **On Paderborn the claim still needs measured speed.**
- **C1b. Estimator rewritten (2026-10-03), against the user's tests A1–A5 and B1–B2 in `tests/`.** Old bugs:
  1. a clean harmonic series was estimated at 1.25×;
  2. white noise got confidence 1.0;
  3. a rule preferred the higher candidate.

  The new `speed.py` scores 1×–3× harmonics (weights 0.5/0.25/0.25) against the spectral noise floor, with no direction preference. Confidence is the weighted share of harmonics above a noise-derived threshold (3.65 × the median magnitude, a 1e-4 per-bin false-alarm rate for Gaussian noise). It was not tuned on real data: the real-data accuracy above was measured once, after the tests passed. Per the plan, no further tuning after Oct 3.
- **C1c. CIs are wide.** CWRU P3 envelope_ratio is 0.46 [0.25, 0.75] with 14 faults. Paderborn P3 is 0.75 [0.58, 0.86] with 29 faults. Report intervals, not points.
- **C2. Two models, one seed.** Logistic regression is the headline; GBDT is the second tier. Each is one seed. Across 5 GBDT seeds, P3 macro-F1 is stable (CWRU sd 0.000, Paderborn sd ≤ 0.003). CWRU **P1 and P2** are not: `full` and `envelope_ratio` swing between 0.82 and 0.99 by seed alone. The swing is entirely the healthy class (5 windows). Don't read feature-set differences at CWRU P1/P2 from a single seed.
- **C2b. Loss curves are descriptive.** The "best stage" marker is chosen on the held-out folds, so it is not a legitimate early-stopping choice. It only shows *that* held-out loss turns up after 1–22 stages under P3. Early stopping would need a validation fault held out *inside* each training fold. Losses are floored at 1e-4 for the log axis. On CWRU the held-out P3 loss leaves out the healthy fold, where the class is absent from training so the loss is infinite. That fold is kept in the macro-F1 curve.
- **C3. No test gate.** `tests/` was empty during this run. The loaders were checked by hand on real files, not by a verifier.
- **C4. P0 is invalid by construction.** Random-window splits appear only to show the collapse. Never report them as results.
- **C5. The P1 caveat.** On datasets with one window per recording (Paderborn, MaFaulDa), leave-one-recording-out equals P0, so P1 isn't run there.

### CWRU

- **C6. One healthy bearing.** Only `97.mat` and `100.mat` survive KNOWN_BAD, and they are one physical identity. Under P3 the healthy fold trains with no healthy data, so healthy is always misclassified. That caps macro-F1 and says nothing about the features.
- **C7. Small n.** 52 files and 14 fault identities, so every P3 number has wide uncertainty. There are no confidence intervals yet; `bootstrap_ci_fault_level` exists but isn't wired in.
- **C8. Deployment view is uninformative on CWRU.** The source is 12 kHz, so a 6 kHz low-pass removes almost nothing. Use Paderborn (64 kHz) for the degradation claim.
- **C9. IR028 and B028 are gone.** 3001–3008 are KNOWN_BAD, which drops the 0.028″ severity.

### Paderborn

- **C10. One corrupt file.** `KA08/N15_M01_F10_KA08_2.mat` fails to parse and is skipped (2,319 of 2,320 files). It may be a download or extraction fault; its hash is recorded.
- **C11. Unknown units.** The documentation doesn't give the units of `vibration_1`. The deployment view's MEMS noise assumes g, so its size relative to the signal is uncertain.
- **C12. Healthy split for artificial → real.** K001–K003 train and K004–K006 test. That is our choice, not a published protocol.

### MaFaulDa

- **C13. Cage faults are counted as outer race** (the existing taxonomy rule). 376 of the 748 `bearing_OR` files (half) are cage faults, and that contaminates the OR class. Split them out or drop them before making any MaFaulDa bearing claim.
- **C14. P3 is impossible** (one rig, one object per class). The only honest MaFaulDa experiment is held-out imbalance severity (C11).
- **C15. Healthy vs imbalance is easy.** Every feature set scores about 0.92–1.00, so C11 doesn't discriminate between feature sets.
- **C16. Single accelerometer channel.** Underhang axial (column 2) is used, even though radial is the usual choice for bearing faults. It was not tuned.
- **C17. Licence not stated** by UFRJ. Cite the authors and don't claim a licence.

### MFPT and transfer

- **C18. The MFPT diagonal is leaky.** There are too few fault identities for P3, so its within-dataset cell uses leave-one-recording-out, and 0.96–1.00 is inflated. Only the off-diagonal cells are honest.
- **C19. Transfer uses 3 classes** (healthy, IR, OR), because Paderborn and MFPT have no ball faults.
- **C20. Transfer windows are 2 s**, set by MFPT's 3 s recordings. That gives 0.5 Hz envelope resolution instead of 0.25 Hz.
- **C21. MFPT is a third-party Kaggle mirror** (`emperorpein/mfpt-fault-datasets`). `mfpt.org` is dead.

### Spec wording to correct

- **C22. §7 energy step.** A coarse grid doesn't *miss* the step, it *overshoots*: the error is ≈ ΔP·dt/2, so dt = 1 s gives 55 J. dt = 1e-3 only just passes (50.005 J).
- **C23. §5 decimation.** Resolution is 1/T whatever the decimation. Decimation only sets Nyquist (≈ 834 Hz).
- **C24. §3.2 seeding.** The tolerances hold with `np.random.default_rng(0)`. Pin the seed.
