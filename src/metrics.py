"""Evaluation metrics implemented from scratch.

References: Rousseeuw (1987) for silhouette; Hubert & Arabie (1985) for
the Adjusted Rand Index. See ../REFERENCES.md for complete citations.
"""
from __future__ import annotations

import math
import numpy as np


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    sq = np.sum(X * X, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
    return np.sqrt(d2)


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(X):
        return float("nan")
    D = pairwise_euclidean(X)
    scores = np.zeros(len(X), dtype=float)
    for i in range(len(X)):
        own = labels[i]
        own_mask = labels == own
        own_count = int(np.sum(own_mask))
        if own_count <= 1:
            scores[i] = 0.0
            continue
        a = float(np.sum(D[i, own_mask]) / (own_count - 1))
        b = math.inf
        for other in unique:
            if other == own:
                continue
            mask = labels == other
            if np.any(mask):
                b = min(b, float(np.mean(D[i, mask])))
        denom = max(a, b)
        scores[i] = 0.0 if denom <= 0 else (b - a) / denom
    return float(np.mean(scores))


def _comb2(n: int) -> float:
    return n * (n - 1) / 2.0


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted Rand Index without sklearn."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    if len(labels_true) != len(labels_pred):
        raise ValueError("label arrays must have the same length")

    classes, class_inv = np.unique(labels_true, return_inverse=True)
    clusters, cluster_inv = np.unique(labels_pred, return_inverse=True)
    contingency = np.zeros((len(classes), len(clusters)), dtype=int)
    np.add.at(contingency, (class_inv, cluster_inv), 1)

    sum_comb = sum(_comb2(int(n)) for n in contingency.ravel())
    sum_rows = sum(_comb2(int(n)) for n in contingency.sum(axis=1))
    sum_cols = sum(_comb2(int(n)) for n in contingency.sum(axis=0))
    total = _comb2(len(labels_true))
    if total == 0:
        return 1.0
    expected = (sum_rows * sum_cols) / total
    max_index = 0.5 * (sum_rows + sum_cols)
    denom = max_index - expected
    if abs(denom) < 1e-15:
        return 1.0 if abs(sum_comb - max_index) < 1e-15 else 0.0
    return float((sum_comb - expected) / denom)


def cluster_centroids(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    return np.vstack([np.mean(X[labels == k], axis=0) for k in unique])
