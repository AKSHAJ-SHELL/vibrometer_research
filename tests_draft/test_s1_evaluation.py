"""Spec §1 — Evaluation. Fixes the train/test baseline bug in c1_results.json.

Targets: vibedge.evaluate, vibedge.models, vibedge.splits, vibedge.experiments
Tolerance: pytest.approx(value, abs=1e-6)
"""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from vibedge.evaluate import (
    fold_macro_f1_is_defined,
    majority_baseline_metrics,
    majority_label,
    pooled_evaluation,
)
from vibedge.experiments import run_c1_synthetic, score_protocol
from vibedge.models import fit_model, predict_folds
from vibedge.splits import make_splits

TOL = dict(abs=1e-6)


def test_majority_baseline_is_fit_on_train_labels_not_test():
    # 1.1 — 1.0 here would mean the baseline read the test labels
    y_train = np.array([0, 0, 1])
    y_test = np.array([1])
    assert majority_label(y_train) == 0
    f1, acc = majority_baseline_metrics(y_test, y_train=y_train, labels=[0, 1])
    assert acc == pytest.approx(0.0, **TOL)
    assert f1 == pytest.approx(0.0, **TOL)
    # the model tier agrees
    clf = fit_model("majority", np.zeros((3, 1)), y_train)
    assert clf.predict(np.zeros((1, 1))).tolist() == [0]


def test_majority_baseline_metrics_known_vector():
    # 1.2 — class0 F1 0.75, class1 F1 0 → mean 0.375
    y = np.array([0, 0, 0, 1, 1])
    f1, acc = majority_baseline_metrics(y, y_train=y)
    assert acc == pytest.approx(0.6, **TOL)
    assert f1 == pytest.approx(0.375, **TOL)


def test_logo_f1_is_pooled_not_fold_averaged():
    # 1.3
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    fold_true = [np.array([t]) for t in y_true]
    fold_pred = [np.array([p]) for p in y_pred]
    ev = pooled_evaluation(fold_true, fold_pred, labels=[0, 1, 2])
    assert ev.macro_f1 == pytest.approx(0.655556, **TOL)
    assert ev.accuracy == pytest.approx(0.666667, **TOL)
    # the wrong answer: mean of per-fold F1 (= accuracy for one-sample folds)
    wrong = np.mean([f1_score(t, p, average="macro", zero_division=0) for t, p in zip(fold_true, fold_pred)])
    assert wrong == pytest.approx(0.666667, **TOL)
    assert ev.macro_f1 != pytest.approx(wrong, **TOL)


def test_train_majority_under_leave_one_out_on_balanced_classes():
    # 1.4 — expected behaviour, not a bug: the held-out class is the training minority
    y = np.array([0, 0, 1, 1, 2, 2])
    rec = [f"r{i}" for i in range(6)]
    labels = ["healthy", "healthy", "imbalance", "imbalance", "misalignment", "misalignment"]
    splits = make_splits(1, recording_ids=rec, fault_ids=rec, labels=labels)
    assert len(splits) == 6
    preds = np.empty(6, dtype=int)
    for test_idx, p in predict_folds("majority", np.zeros((6, 1)), y, splits):
        preds[test_idx] = p
    assert preds.tolist() == [1, 1, 0, 0, 0, 0]  # ties → lowest label
    ev = pooled_evaluation([y[s.test_idx] for s in splits], [preds[s.test_idx] for s in splits], labels=[0, 1, 2])
    assert ev.macro_f1 == pytest.approx(0.0, **TOL)
    _, pooled = score_protocol(np.zeros((6, 1)), y, splits, [0, 1, 2], model_name="majority")
    assert pooled["majority_baseline_f1"] == pytest.approx(0.0, **TOL)


def test_single_sample_folds_do_not_report_their_own_macro_f1(tmp_path):
    # 1.5
    assert not fold_macro_f1_is_defined(np.array([1]))
    assert not fold_macro_f1_is_defined(np.array([2, 2, 2]))
    assert fold_macro_f1_is_defined(np.array([0, 1]))

    result = run_c1_synthetic(n_per_class=2, freeze_dir=tmp_path, protocols=[3])
    fold_rows = [r for r in result["rows"] if r["fold"] != "pooled"]
    pooled = [r for r in result["rows"] if r["fold"] == "pooled"]
    assert fold_rows and pooled
    one_sample = [r for r in fold_rows if r["n_test"] == 1]
    assert one_sample, "synthetic P3 should have one-sample folds"
    for r in one_sample:
        assert r["macro_f1"] is None, r
    for r in pooled:
        assert isinstance(r["macro_f1"], float)
