# vibedge — results handoff

_Generated 2026-10-04T06:20:40+00:00 by `scripts/make_handoff.py` from `results/real/*.json`. Every number below is read from those files; regenerate with `make handoff` rather than editing by hand._

Provenance recorded 2026-10-04T06:20:40+00:00 (written by `make reproduce` after its run) · git: 671f53578811ec46f2a76feb119373edc1b572bc · Python 3.12.12 · data manifest sha256 `8caca47c7297dfe8…`

## Headline — pooled macro-F1 with 95% fault-level CI

Honest splits only (P0 random windows is invalid by design and excluded). Δ is a paired bootstrap over the same fault resamples. Shaft speed: measured (oracle) — see the next section.

| Split | Model | envelope_ratio | time_only | Δ (envelope − time) | CWRU excl. healthy: envelope / time |
|---|---|---|---|---|---|
| CWRU · P3 leave-one-fault-out | logreg | 0.457 [0.25, 0.75] | 0.298 [0.15, 0.53] | +0.159 [-0.06, +0.40] | 0.609 [0.32, 0.82] / 0.397 [0.19, 0.59] |
| CWRU · P4 leave-one-severity-out | logreg | 0.681 [0.46, 0.84] | 0.237 [0.08, 0.34] | +0.444 [+0.28, +0.61] | 0.658 [0.41, 0.87] / 0.205 [0.02, 0.34] |
| Paderborn · P3 leave-one-bearing-out | logreg | 0.745 [0.58, 0.86] | 0.506 [0.37, 0.61] | +0.239 [+0.15, +0.33] | — |
| Paderborn · P3b artificial → real damage | logreg | 0.553 [0.31, 0.78] | 0.381 [0.15, 0.57] | +0.172 [+0.06, +0.30] | — |
| CWRU · P3 leave-one-fault-out | GBDT | 0.505 [0.25, 0.85] | 0.311 [0.15, 0.55] | +0.195 [-0.09, +0.47] | 0.674 [0.33, 0.91] / 0.414 [0.20, 0.64] |
| CWRU · P4 leave-one-severity-out | GBDT | 0.748 [0.48, 0.91] | 0.432 [0.13, 0.55] | +0.315 [+0.10, +0.61] | 0.700 [0.40, 0.92] / 0.243 [0.07, 0.40] |
| Paderborn · P3 leave-one-bearing-out | GBDT | 0.700 [0.55, 0.81] | 0.657 [0.51, 0.77] | +0.043 [-0.04, +0.11] | — |
| Paderborn · P3b artificial → real damage | GBDT | 0.587 [0.35, 0.81] | 0.388 [0.19, 0.58] | +0.199 [+0.08, +0.35] | — |

## Shaft-speed accuracy per speed source

| Dataset | Speed source | within 2% | within 5% | locked to 2× | median rel. error | speed features used | Δ envelope − time (first honest split) |
|---|---|---|---|---|---|---|---|
| cwru | oracle | 100% | 100% | 0% | 0.0% | 100% | P3: +0.159 [-0.06, +0.40] |
| cwru | estimated | 12% | 12% | 67% | 99.9% | 100% | P3: +0.204 [-0.05, +0.43] |
| cwru | prior | 100% | 100% | 0% | 0.0% | 100% | P3: +0.159 [-0.06, +0.40] |
| paderborn | oracle | 100% | 100% | 0% | 0.0% | 100% | P3: +0.239 [+0.15, +0.33] |
| paderborn | estimated | 38% | 41% | 12% | 31.3% | 40% | P3: +0.050 [-0.02, +0.12] |
| paderborn | prior | 57% | 65% | 0% | 1.6% | 37% | P3: +0.042 [-0.03, +0.11] |

## Cost column (EMC²)

Measured on one 4 s window of `105.mat` at 12000 Hz; models fit on all CWRU windows. Host-CPU timing is a relative ranking, not MCU latency.

| Feature set | Features | Bytes (float32) | Logreg params (model + scaler) | GBDT params (model + scaler) | Extract time / window |
|---|---|---|---|---|---|
| time_only | 17 | 68 | 106 (72 + 34) | 10320 (10286 + 34) | 1.17 ms |
| envelope_ratio | 39 | 156 | 238 (160 + 78) | 5372 (5294 + 78) | 3.66 ms |
| full | 40 | 160 | 244 (164 + 80) | 5380 (5300 + 80) | 4.14 ms |

## Figures

- `figures/real/C1_ladder_real.png` — Pooled macro-F1 across the split ladder (P0 → P4) for each feature set, CWRU and Paderborn.
- `figures/real/speed_ablation.png` — The same honest splits with measured, estimated and rated-speed-prior shaft speed, with 95% fault-level CIs.
- `figures/real/C10_transfer_real.png` — Cross-dataset transfer (train on row, test on column) for time-only vs envelope features.
- `figures/real/deployment_degradation_real.png` — P3 score before and after the simulated deployment sensor (6 kHz LPF, 26.7 kSPS, MEMS noise).
- `figures/real/A3_envelope_cwru.png` — Real CWRU envelope spectra with BPFO / BPFI / 2×BSF marked — the physics sanity check.

## Top limitations

_Quoted verbatim from `adjustments_and_cliffs.md`._

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
- **C6. One healthy bearing.** Only `97.mat` and `100.mat` survive KNOWN_BAD, and they are one physical identity. Under P3 the healthy fold trains with no healthy data, so healthy is always misclassified. That caps macro-F1 and says nothing about the features.
- **C2. Two models, one seed.** Logistic regression is the headline; GBDT is the second tier. Each is one seed. Across 5 GBDT seeds, P3 macro-F1 is stable (CWRU sd 0.000, Paderborn sd ≤ 0.003). CWRU **P1 and P2** are not: `full` and `envelope_ratio` swing between 0.82 and 0.99 by seed alone. The swing is entirely the healthy class (5 windows). Don't read feature-set differences at CWRU P1/P2 from a single seed.
- **C3. No test gate.** `tests/` was empty during this run. The loaders were checked by hand on real files, not by a verifier.

## AI-use disclosure

<!-- To be written by the author. -->
