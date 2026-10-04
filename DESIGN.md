# vibedge — On-Device Bearing and Imbalance Fault Detection (Software v2)

Leakage-safe evaluation of physics-grounded vibration features for edge deployment.

## What this is

Software pipeline that:

1. Loads public vibration datasets (MaFaulDa primary; CWRU/MFPT/Paderborn for transfer).
2. Applies a **deployment view** (6 kHz LPF, 26.7 kSPS, MEMS noise) so training matches the target sensor.
3. Estimates shaft speed from vibration (no tachometer at deploy time).
4. Extracts a two-timescale feature vector (fast broadband + slow envelope).
5. Evaluates under a leakage ladder (protocols 0–5) with frozen split manifests.
6. Tests the hypothesis: **envelope-ratio features degrade less than time-domain features as split rigor increases.**

## Explicit limitations (do not paper over)

- **Imbalance** can only be evaluated with held-out severity on MaFaulDa (one rig). Fault-identity and cross-dataset splits are impossible for imbalance on public data.
- **Speed-dependent features** require a confident `f_r` estimate; otherwise the speed-free subset is used.
- **Bearing geometry** (`n, d, D, φ`) must be provided per machine / dataset; kinematic frequencies are not universal constants.
- Phase / multi-sensor discrimination of misalignment subtypes is out of scope (single accelerometer).

## Quick start

```bash
make setup
make test
make figures   # synthetic smoke path — no downloads required
```

## Layout

See repository tree under `src/vibedge/`. Build order and rationale are in the project plan.

## Evaluation methods (state these in the paper)

- **Majority baseline is fit on each fold's training labels** and scored on
  its test labels. (Before 2026-09-29 it was scored in-sample on the test
  labels, which gave `1.0` on every single-class fold of `c1_results.json`.)
- **Leave-one-group-out scores are pooled**: every fold's predictions are
  concatenated and macro-F1 is computed once (rows with `fold: "pooled"`).
  Averaging per-fold macro-F1 is wrong for small folds; with one test sample
  per fold it collapses to accuracy. Per-fold macro-F1 is reported as `None`
  when a fold has < 2 test samples or < 2 classes.
- **Train-majority under leave-one-out on balanced classes is always wrong.**
  Holding out one sample makes its class the training minority, so the
  baseline predicts another class on every fold and its pooled macro-F1 is 0.
  This is expected, not a bug; we report it anyway.
- **Macro-F1 averages over the classes present in the dataset** (4 on CWRU),
  not all 6 taxonomy classes — absent classes would otherwise count as F1 = 0.
- **Kurtosis is Pearson** (Gaussian = 3, unit sine = 1.5), from population
  moments. scipy's default is Fisher (excess, Gaussian = 0).
- **No recording on both sides of a split** for any protocol ≥ 1:
  `splits.make_splits` raises `LeakageError` naming the offending recordings.
  Healthy recordings in protocol 4 are dealt to train/test per recording,
  never per window.

- **CWRU also reports `macro_f1_excl_healthy` on P3/P4 pooled rows.** CWRU
  has a single healthy bearing (97.mat + 100.mat after KNOWN_BAD). Under
  leave-one-fault-out the healthy fold trains with no healthy data, so healthy
  F1 is 0 by construction, whatever the features. That caps macro-F1 at 3/4 of
  the fault-class score. `macro_f1_excl_healthy` averages F1 over IR, OR and
  ball only. Healthy windows stay in the data, so a healthy window predicted as
  a fault still counts against that fault's precision. It carries its own
  fault-level `ci95`. Report both numbers; the full macro-F1 stays the headline.

## Dataset handling notes

- **CWRU** downloads are numbered; `datasets/cwru_catalog.py` (scraped from the
  CWRU pages) maps number → subset and fault. Fault identity ignores load and
  sampling rate, and includes the outer-race position (@3/@6/@12). On the 12k
  drive-end set + normal baseline: 52 usable files, 14 fault identities.
  Healthy is a single identity (one bearing), so its protocol-3 fold trains
  without healthy data.
- **CWRU KNOWN_BAD** gained 174.mat and 3002–3008.mat after inspecting the
  2026-09-29 download: none contain an array named after their file number
  (same defect as 3001.mat). The loader also picks the array matching the file
  number (99.mat also ships X098_* arrays).
- **Paderborn** `KB*` bearings are combined inner+outer damage, not ball
  faults; they are excluded from the 6-class taxonomy. Geometry is the 6203
  (n = 8, d = 6.75 mm, D = 28.55 mm).
- **MFPT** geometry is n = 8, d = 0.235 in, D = 1.245 in with the per-file shaft rate.
  Fault identity is per fault type (conservative), so protocol 3 is infeasible.
- **Shaft speed from metadata**: CWRU RPM fields and MaFaulDa filenames give
  the measured shaft rate. Using them for order/envelope features is oracle
  speed; the deploy-time path estimates speed from vibration (`speed.py`).
