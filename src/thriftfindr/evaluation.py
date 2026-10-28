"""Metrics shared by the frozen-feature baselines and the fine-tuned models."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score


def classification_metrics(y_true, y_pred, n_classes: int, scores=None, groups=None) -> dict:
    """Accuracy, macro-F1, balanced accuracy, top-3 (if class scores are given), confusion matrix,
    and a 95% bootstrap interval for accuracy that resamples whole groups (images of one product
    or runway show are correlated, so resampling images would understate the uncertainty)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.arange(n_classes)
    out = {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if scores is not None and n_classes > 3:
        out["top3_accuracy"] = float(top_k_accuracy_score(y_true, scores, k=3, labels=labels))
    if groups is not None:
        out["accuracy_ci95"] = group_bootstrap_ci(y_true == y_pred, groups)
    return out


def era_accuracy_by_brand(y_brand, y_era, era_correct, brand_names) -> dict:
    """Era accuracy per true brand; None where a brand's test images all share one era, since accuracy
    there could come from recognising the brand alone."""
    out = {}
    for i, b in enumerate(brand_names):
        m = np.asarray(y_brand) == i
        multi_era = m.any() and len(np.unique(np.asarray(y_era)[m])) > 1
        out[b] = float(np.asarray(era_correct)[m].mean()) if multi_era else None
    return out


def group_bootstrap_ci(correct, groups, n_boot: int = 2000, seed: int = 0) -> list[float]:
    correct = np.asarray(correct, dtype=float)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    hits = np.bincount(inverse, weights=correct)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sizes), size=(n_boot, len(sizes)))
    acc = hits[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]


def price_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ae = np.abs(y_pred - y_true)
    ape = ae / y_true
    return {
        "n": int(len(y_true)),
        "mae": float(ae.mean()),
        "median_ae": float(np.median(ae)),
        "mdape_pct": float(100 * np.median(ape)),
        "within_25pct": float((ape <= 0.25).mean()),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }
