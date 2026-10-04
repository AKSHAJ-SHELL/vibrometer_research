"""Model tiers: majority baseline, logistic regression (T1), GBDT (T2).

T3 1-D CNN is optional / host-side; stubbed for interface parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ModelName = Literal["majority", "logreg", "gbdt", "cnn_stub"]


@dataclass
class FittedModel:
    name: str
    model: Any
    classes_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.name == "majority":
            return np.full(len(X), self.model, dtype=np.int64)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.name == "majority":
            n_classes = len(self.classes_)
            proba = np.zeros((len(X), n_classes))
            # map majority label to column
            idx = int(np.where(self.classes_ == self.model)[0][0])
            proba[:, idx] = 1.0
            return proba
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None


class MajorityBaseline:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityBaseline":
        vals, counts = np.unique(y, return_counts=True)
        self.label_ = int(vals[np.argmax(counts)])
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.label_, dtype=np.int64)


def fit_model(
    name: ModelName,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 0,
    gbdt_params: dict | None = None,
) -> FittedModel:
    """Fit standardisation on training fold only (no global scaling leakage)."""
    y_train = np.asarray(y_train, dtype=np.int64)
    if name == "majority":
        m = MajorityBaseline().fit(X_train, y_train)
        return FittedModel("majority", m.label_, m.classes_)

    if name == "logreg":
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        return FittedModel("logreg", pipe, pipe.named_steps["clf"].classes_)

    if name == "gbdt":
        params = {
            "max_depth": 4,
            "n_estimators": 100,
            "random_state": seed,
            "learning_rate": 0.1,
        }
        if gbdt_params:
            params.update(gbdt_params)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(**params)),
            ]
        )
        pipe.fit(X_train, y_train)
        return FittedModel("gbdt", pipe, pipe.named_steps["clf"].classes_)

    if name == "cnn_stub":
        # Host-side placeholder: falls back to logreg so scripts don't break
        return fit_model("logreg", X_train, y_train, seed=seed)

    raise ValueError(f"Unknown model: {name}")


def predict_folds(
    name: ModelName,
    X: np.ndarray,
    y: np.ndarray,
    splits: list,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fit on each split's train indices, predict its test indices.

    Returns [(test_idx, y_pred)] per fold. Every model — including the
    majority baseline — sees only y[train_idx].
    """
    y = np.asarray(y, dtype=np.int64)
    out = []
    for split in splits:
        clf = fit_model(name, X[split.train_idx], y[split.train_idx], seed=seed)
        out.append((split.test_idx, np.asarray(clf.predict(X[split.test_idx]), dtype=np.int64)))
    return out


def model_param_count(fitted: FittedModel) -> dict[str, int]:
    """Parameters that must ship to the device for inference.

    ``model``: learned weights (logreg coef+intercept; GBDT one threshold +
    feature index per split node and one value per leaf, per class-tree).
    ``preprocessing``: StandardScaler mean + scale.
    """
    if fitted.name == "majority":
        return {"model": 1, "preprocessing": 0, "total": 1}
    pipe = fitted.model
    scaler = pipe.named_steps.get("scaler")
    pre = int(scaler.mean_.size + scaler.scale_.size) if scaler is not None else 0
    clf = pipe.named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        m = int(clf.coef_.size + clf.intercept_.size)
    elif isinstance(clf, GradientBoostingClassifier):
        m = 0
        for est in clf.estimators_.ravel():
            t = est.tree_
            leaves = int(np.sum(t.children_left == -1))
            splits = int(t.node_count - leaves)
            m += 2 * splits + leaves
        prior = getattr(clf.init_, "class_prior_", None)  # initial raw prediction
        m += int(np.size(prior)) if prior is not None else 0
    else:
        raise TypeError(f"param count not implemented for {type(clf).__name__}")
    return {"model": m, "preprocessing": pre, "total": m + pre}
