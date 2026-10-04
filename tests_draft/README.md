# tests_draft — draft verifier and test documentation

Implemented draft verifier for the vibedge test spec: 35 tests, one file per
spec section. `tests/` is untouched.

**Written by the agent, at your request, 2026-09-29.** The same agent wrote the
`src/` code under test, so a pass here is not independent verification. Review
each file against the spec before you rely on it or move it into `tests/`.

Running:

    make check                       # tests/ + tests_draft/ in one run
    make test-draft                  # tests_draft/ only
    python -m pytest                 # both (testpaths in pyproject.toml)

- `draft_paths.py` holds the shared paths and the `real_data` marker. It isn't
  named `conftest.py`, so it won't clash with a `conftest.py` you add to `tests/`.
- The two `real_data` tests skip when `data/cwru/` is absent.

Planted-bug check: each of these, put into a scratch copy of `src/`, makes at
least one test fail, while the clean code passes all 35.
- the baseline reads the test labels (§1.1);
- Fisher kurtosis with bias correction (§3.1);
- healthy windows split per window, with the leakage guard off (§2.1);
- load baked into the CWRU fault identity (§2.2 and §2.3);
- fold-averaged instead of pooled F1 (§1.3);
- KNOWN_BAD ignored (§2.4).

Spec priority: §1 → §2 → §3.5 → §4 → the rest. §1 and §2 decide whether
the paper's numbers are real; §3 decides whether the features are.

The expected numbers below come from your spec. I recomputed each one with
numpy/scipy/sklearn, independently of `src/`, and all of them match.

---

## §1 Evaluation — `test_s1_evaluation.py`

| Test | Input | Expected | Fails if |
|---|---|---|---|
| 1.1 baseline fit on train | train `[0,0,1]`, test `[1]`, `labels=[0,1]` | acc 0.0, macro-F1 0.0 | baseline returns 1.0, i.e. it read the test labels (the old `c1_results.json` bug) |
| 1.2 known vector | `y=[0,0,0,1,1]`, baseline fit on `y` | acc 0.6, macro-F1 0.375 | F1 ≠ mean(0.75, 0) |
| 1.3 pooled LOGO | 6 one-sample folds, `y_true=[0,0,1,1,2,2]`, `y_pred=[0,1,1,1,2,0]` | pooled macro-F1 0.655556, acc 0.666667 | returns 0.666667, the mean of per-fold F1s |
| 1.4 LOO majority | same `y_true`, leave-one-out, ties → lowest label | preds `[1,1,0,0,0,0]`, pooled F1 0.0 | any prediction matches the held-out label |
| 1.5 single-sample folds | `run_c1_synthetic(protocols=[3])` | per-fold `macro_f1 is None` when `n_test == 1`; a `fold == "pooled"` row exists | a one-sample fold reports its own macro-F1 |

APIs you'll call:

- `evaluate.majority_baseline_metrics(y_true, y_train=None, labels=None) -> (f1, acc)`: fits on `y_train`. Without `y_train` it warns and scores in-sample.
- `evaluate.majority_label(y_train) -> int`: ties go to the lowest label.
- `evaluate.pooled_evaluation(fold_true, fold_pred, labels=None, fold_baseline_pred=None) -> EvalResult`: `.macro_f1`, `.accuracy`.
- `evaluate.fold_macro_f1_is_defined(y_true) -> bool`: needs ≥ 2 samples and ≥ 2 classes.
- `models.predict_folds(name, X, y, splits) -> [(test_idx, y_pred), ...]`
- `experiments.score_protocol(X, y, splits, labels, model_name=...) -> (fold_rows, pooled_row)`
- `experiments.run_c1_synthetic(n_per_class, freeze_dir, protocols) -> {"rows": [...]}`

Paper note for 1.4: on balanced data, holding out a sample makes its class the
training minority, so the train-majority baseline is always wrong. Report it
anyway, and say so in the methods section (DESIGN.md already does).

---

## §2 Leakage — `test_s2_leakage.py`

| Test | Setup | Expected | Fails if |
|---|---|---|---|
| 2.1 `test_no_recording_overlap` | 3 recordings × 10 windows (`r0,r1,r2`), protocols 1–5 | recording ids in train ∩ test = ∅ for every fold | any id is shared; the message must name it |
| 2.1 negative case | hand-built split with `r1` on both sides | `LeakageError` whose message contains `r1` | no error is raised |
| 2.2 identity | `IR007_0..3.mat` | all four → `cwru_IR_0.007` | load leaks into the id |
| 2.2 same fold | `IR007_0..3` plus other faults, protocol 3 | all four land in one test fold | any notch has one load in train and another in test |
| 2.3 fold count | synthetic files, then real CWRU | `n_folds == n_fault_identities < n_files` (real: 14 < 52) | `n_folds == n_files` |
| 2.4 KNOWN_BAD | numbered files including `191.mat` | `191.mat` is never loaded or used in any split | a KNOWN_BAD file appears |

APIs you'll call:

- `splits.make_splits(protocol, *, recording_ids, fault_ids, labels, loads=None, severities=None, datasets=None) -> [SplitIndices]`

  | Protocol | Folds by | Needs |
  |---|---|---|
  | 1 | leave-one-recording-out | — |
  | 2 | leave-one-load-out | `loads` |
  | 3 | leave-one-fault-identity-out | — |
  | 4 | leave-one-severity-out (healthy recordings assigned to folds whole) | `severities` + a `"healthy"` label |
  | 5 | leave-one-dataset-out | `datasets` |

- `splits.check_no_group_overlap(split, groups, what="recording")`: raises `LeakageError`.
- `splits.SplitIndices(protocol, train_idx, test_idx, fold=0, meta={})`
- `windowing.make_windows([(signal, SampleMeta), ...], window_s, overlap=0.0)`: each window keeps `meta.file_id` as its recording id.
- `windowing.meta_columns(windows)`: returns the keyword arguments `make_splits` expects, so use `make_splits(p, **cols)`.
- `datasets.cwru.load_cwru_directory(root, subsets=("normal","12k_DE"), target_fs=None)`: pass `subsets=None` for tmp dirs.
- `datasets.cwru.KNOWN_BAD`: dict of filename → reason.

Writing fake CWRU `.mat` files with `scipy.io.savemat`:

- **Fault-named** (`IR007_0.mat`): any key containing `DE` works, e.g. `{"X000_DE_time": signal}`.
- **Numbered** (`191.mat`): the key must match the number, e.g. `{"X191_DE_time": signal, "X191RPM": 1797}`. The loader only reads the array named after the file, because `99.mat` also contains file 98's data.

Real-data tests should use `pytest.mark.skipif(not Path("data/cwru").exists(), ...)`.

---

## §3 Signal features — `test_s3_signal_features.py`

| Test | Signal | Expected |
|---|---|---|
| 3.1 | unit sine, fs = 100 000, 1 s, 50 Hz | RMS 0.707107, crest 1.414214, kurtosis 1.5 (Pearson) |
| 3.2 | `np.random.default_rng(0).standard_normal(1_000_000)` | RMS ≈ 1.00 (abs 0.01), Pearson ≈ 3.0 / Fisher ≈ 0.0 (abs 0.02) |
| 3.3 | 50 Hz, fs = 1000, N = 4096 | peak bin 205, bin width 0.244141 Hz, amplitude 0.935489 (abs 1e-4) |
| 3.4 | same | Σx² = spectral energy = 2048.0 (rel 1e-9) |
| 3.5 | `(1+0.5cos(2π·100t))·sin(2π·3000t)`, fs = 12 000, 2 s | envelope peak at 100.0 Hz, resolution 0.5 Hz |

APIs you'll call:

- `features.time_domain.time_domain_features(x) -> dict`: keys `rms`, `crest_factor`, `kurtosis`, …
- `features.time_domain.KURTOSIS_CONVENTION == "pearson"`: population moments.
- `features.spectral.single_sided_amplitude(x, fs) -> (freqs, amp)`: `2·|X[k]|/N`, rectangular window.
- `features.spectral.spectral_energy(x) -> float`: `(1/N)·Σ|FFT(x)|²`.
- `features.envelope.envelope(x)`: returns `|hilbert(x)|`.
- `features.envelope.compute_envelope_spectrum(x, fs, band=(2000, 5000), decim=16) -> .freqs, .mag`

Notes:

- The kurtosis convention matters. `scipy.stats.kurtosis` defaults to Fisher, while the features use Pearson. The old code also used the bias-corrected estimator, which gives −1.500015 for the sine and fails abs 1e-6.
- Pin the §3.2 seed, or the tolerances can fail by chance.

---

## §4 Bearing frequencies — `test_s4_bearing_freqs.py`

API: `features.bearing.bearing_freqs(fr, CWRU_6205_DE) -> .bpfo .bpfi .ftf .bsf .ball_fault`

| | × fr | at 1797 rpm (fr = 29.95 Hz) |
|---|---|---|
| BPFO | 3.584776 | 107.364 Hz |
| BPFI | 5.415224 | 162.186 Hz |
| FTF | 0.398308 | — |
| BSF | 2.356722 | — |
| BPFO + BPFI | 9 (= n) | 269.55 Hz |

---

## §5 Deployment view — `test_s5_deployment.py`

API: `features.envelope.envelope_resolution_hz(26_700, 4.0, 16)` → 0.25 Hz, which must be < 0.5 Hz.

Resolution is `1/T` whatever the decimation. Decimating by 16 only moves
Nyquist to ≈ 834 Hz, so the 0.25 Hz assertion never tests the decimation.
The second stub checks that the highest band in use (≈ 3·BPFI + fr) is still
below Nyquist.

---

## §6 Cost column — `test_s6_cost.py`

API: `cost.cost_table(x_window, fs, fr_hz, X_by_set=None, y=None) -> [row per feature set]`

| Row key | Meaning |
|---|---|
| `n_features` | feature count (`full` must be 40) |
| `feature_bytes_float32` | 4 × `n_features` (`full` → 160) |
| `params_logreg`, `params_gbdt` | `{"model", "preprocessing", "total"}`; only present when `X_by_set` and `y` are given |
| `extract_s_per_window` | host-CPU median time per window (relative ranking only, not MCU latency) |

The stale-table check compares `results/features_synthetic.json["n_features"]` against 40.

---

## §7 Energy integration — `test_s7_energy.py`

API: `energy.integrate_energy_j(t_s, p_w) -> J` (trapezoid rule). It raises
`ValueError` on unequal lengths, non-finite samples, or timestamps that
aren't strictly increasing.

| Trace | Expected |
|---|---|
| 5 W constant, 10 s | 50 J |
| ramp P(t) = t, 0–10 s | 50 J |
| step 0 W → 10 W at 5 s, dt ≤ 1e-3 | 50 J (abs 0.01) |
| same step, dt = 1 s | 55 J, so assert \|E − 50\| > 0.01 |

The trapezoid rule doesn't *miss* a step, it *smears* it: the error is
≈ ΔP·dt/2. That's why dt = 1e-3 only just passes (50.005 J) and dt = 1 s
gives 55 J. `energy.max_step_error_j(ΔP, dt)` returns that bound.
