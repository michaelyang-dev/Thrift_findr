"""Price estimation from image features.

Retrieval (kNN): predict the similarity-weighted geometric mean price of the k most similar training
items, i.e. "what do the closest matching listings cost". Compared against a ridge regression on
log price and against median-price-per-brand lookups (with the true or the predicted brand).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from thriftfindr.baselines import KNN_TEMPERATURE, _normalize

PRICE_KS = (5, 10, 20, 50, 100)
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def _mdape(y_true, y_pred) -> float:
    return float(np.median(np.abs(y_pred - y_true) / y_true))


def knn_price(X_train, p_train, X_query, k: int, chunk: int = 2048) -> np.ndarray:
    A, Q, logp = _normalize(X_train), _normalize(X_query), np.log(p_train)
    k = min(k, len(A))
    out = np.empty(len(Q))
    for s in range(0, len(Q), chunk):
        sims = Q[s:s + chunk] @ A.T
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k] if k < sims.shape[1] else np.argsort(-sims, axis=1)
        w = np.exp(np.take_along_axis(sims, idx, axis=1) / KNN_TEMPERATURE)
        out[s:s + chunk] = np.exp((w * logp[idx]).sum(1) / w.sum(1))
    return out


def fit_knn_price(X_train, p_train, X_val, p_val, ks=PRICE_KS) -> int:
    return min(ks, key=lambda k: _mdape(p_val, knn_price(X_train, p_train, X_val, k)))


def fit_ridge_price(X_train, p_train, X_val, p_val, alphas=RIDGE_ALPHAS):
    scaler = StandardScaler().fit(X_train)
    Xt, Xv = scaler.transform(X_train), scaler.transform(X_val)
    fits = {a: Ridge(alpha=a).fit(Xt, np.log(p_train)) for a in alphas}
    alpha = min(alphas, key=lambda a: _mdape(p_val, np.exp(fits[a].predict(Xv))))
    return alpha, lambda X: np.exp(fits[alpha].predict(scaler.transform(X)))


def median_lookup(train_keys, p_train, query_keys) -> np.ndarray:
    """Median training price per key (e.g. brand), falling back to the global median."""
    medians = pd.Series(p_train).groupby(np.asarray(train_keys)).median()
    return pd.Series(np.asarray(query_keys)).map(medians).fillna(float(np.median(p_train))).to_numpy()
