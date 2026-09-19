"""Métriques d'évaluation calculées directement avec NumPy / Python.

Silhouette : Rousseeuw (1987).
Adjusted Rand Index : Hubert & Arabie (1985).
"""
from __future__ import annotations
import math
import numpy as np


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    aa = np.sum(X * X, axis=1)[:, None]
    d2 = np.maximum(aa + aa.T - 2.0 * (X @ X.T), 0.0)
    return np.sqrt(d2)


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(X):
        return float("nan")
    D = pairwise_euclidean(X)
    values = []
    for i in range(len(X)):
        own = labels == labels[i]
        own[i] = False
        a = float(np.mean(D[i, own])) if np.any(own) else 0.0
        b = np.inf
        for c in unique:
            if c == labels[i]:
                continue
            mask = labels == c
            if np.any(mask):
                b = min(b, float(np.mean(D[i, mask])))
        denom = max(a, b)
        values.append(0.0 if denom == 0 or not np.isfinite(denom) else (b - a) / denom)
    return float(np.mean(values))


def _comb2(n: int) -> int:
    return n * (n - 1) // 2


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    y = np.asarray(labels_true)
    z = np.asarray(labels_pred)
    if len(y) != len(z):
        raise ValueError("Les deux vecteurs d'étiquettes doivent avoir la même longueur.")
    true_vals, true_inv = np.unique(y, return_inverse=True)
    pred_vals, pred_inv = np.unique(z, return_inverse=True)
    table = np.zeros((len(true_vals), len(pred_vals)), dtype=int)
    np.add.at(table, (true_inv, pred_inv), 1)

    sum_cells = sum(_comb2(int(v)) for v in table.ravel())
    sum_rows = sum(_comb2(int(v)) for v in table.sum(axis=1))
    sum_cols = sum(_comb2(int(v)) for v in table.sum(axis=0))
    total = _comb2(len(y))
    if total == 0:
        return 1.0
    expected = (sum_rows * sum_cols) / total
    maximum = 0.5 * (sum_rows + sum_cols)
    denom = maximum - expected
    if math.isclose(denom, 0.0):
        return 1.0 if np.array_equal(y, z) else 0.0
    return float((sum_cells - expected) / denom)
