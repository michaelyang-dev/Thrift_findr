"""Frozen-feature baselines: a logistic-regression linear probe and a cosine k-nearest-neighbour vote.

Hyperparameters (C for the probe, k for kNN) are chosen on the validation split by macro-F1; the
test split is touched once, for the final numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

PROBE_CS = (0.01, 0.1, 1.0, 10.0)
KNN_KS = (1, 5, 10, 20, 50)
KNN_TEMPERATURE = 0.07  # similarity-weighted vote, as in the DINO kNN evaluation


@dataclass
class Fitted:
    method: str
    param: float
    val_macro_f1: float
    model: object = None


def _macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def fit_linear_probe(X_train, y_train, X_val, y_val, Cs=PROBE_CS, class_weight="balanced") -> Fitted:
    scaler = StandardScaler().fit(X_train)
    Xt, Xv = scaler.transform(X_train), scaler.transform(X_val)
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight=class_weight).fit(Xt, y_train)
        score = _macro_f1(y_val, clf.predict(Xv))
        if best is None or score > best.val_macro_f1:
            best = Fitted("linear_probe", C, score, (scaler, clf))
    return best


def probe_scores(fitted: Fitted, X) -> np.ndarray:
    scaler, clf = fitted.model
    return clf.predict_proba(scaler.transform(X))


def _normalize(X) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)


def knn_scores(X_train, y_train, X_query, k: int, n_classes: int, chunk: int = 2048) -> np.ndarray:
    """Per-class vote mass from the k most cosine-similar training images, weighted by exp(sim / T)."""
    A, Q = _normalize(X_train), _normalize(X_query)
    out = np.zeros((len(Q), n_classes))
    for s in range(0, len(Q), chunk):
        sims = Q[s:s + chunk] @ A.T
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k] if k < sims.shape[1] else np.argsort(-sims, axis=1)
        w = np.exp(np.take_along_axis(sims, idx, axis=1) / KNN_TEMPERATURE)
        rows = np.repeat(np.arange(len(idx)), idx.shape[1])
        np.add.at(out[s:s + chunk], (rows, y_train[idx].ravel()), w.ravel())
    return out / out.sum(axis=1, keepdims=True)


def fit_knn(X_train, y_train, X_val, y_val, n_classes: int, ks=KNN_KS) -> Fitted:
    best = None
    for k in ks:
        score = _macro_f1(y_val, knn_scores(X_train, y_train, X_val, k, n_classes).argmax(1))
        if best is None or score > best.val_macro_f1:
            best = Fitted("knn", k, score, (X_train, y_train, n_classes))
    return best


def predict_scores(fitted: Fitted, X) -> np.ndarray:
    if fitted.method == "linear_probe":
        return probe_scores(fitted, X)
    X_train, y_train, n_classes = fitted.model
    return knn_scores(X_train, y_train, X, int(fitted.param), n_classes)
