# Superseded split manifests

Nothing here was deleted or edited. Files were **moved**, byte-for-byte, so
the pre-registration record stays complete.

## cwru/ — 74 files, moved 2026-09-30

These hash-named manifests (`protocolN_foldM_<sha12>.json`) were written by the
first real-data runner, `experiments.run_c1_cwru`. The current runner
(`real_experiments.run_c1`) writes the same splits to
`configs/frozen/cwru/protocolN_foldM.json`.

Before the move, all 74 were checked against the current files: each one has
an identical `train_idx` / `test_idx` pair among the new-named files. They are
duplicates of the current pre-registration, not a different one.

## Not moved

The hash-named files at the top level of `configs/frozen/` are the **current**
synthetic pre-registration. `scripts/run_experiment.py --require-frozen` reads
them (`verify_manifest` only scans the top level of `configs/frozen/`), so they
stay where they are.
