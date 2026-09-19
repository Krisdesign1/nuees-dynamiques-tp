"""Data loading, scaling and synthetic demo datasets."""
from __future__ import annotations

import numpy as np


def standardize(X: np.ndarray):
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std, mean, std


def make_demo_dataset(n_per_cluster: int = 120, random_state: int = 7):
    """Three clusters with different geometry, built only with NumPy."""
    rng = np.random.default_rng(random_state)

    # Curved cluster.
    t1 = rng.uniform(-2.6, 2.6, size=n_per_cluster)
    c1 = np.column_stack([
        t1 - 3.0,
        0.45 * (t1 ** 2) + rng.normal(0, 0.22, size=n_per_cluster) - 1.5,
    ])

    # Elongated diagonal cluster.
    t2 = rng.normal(0, 1.25, size=n_per_cluster)
    c2 = np.column_stack([
        1.2 + 1.8 * t2 + rng.normal(0, 0.18, size=n_per_cluster),
        1.7 + 0.55 * t2 + rng.normal(0, 0.20, size=n_per_cluster),
    ])

    # Compact-ish anisotropic cluster.
    z = rng.normal(size=(n_per_cluster, 2))
    A = np.array([[0.55, 0.18], [0.0, 0.9]])
    c3 = z @ A.T + np.array([4.2, -2.3])

    X = np.vstack([c1, c2, c3])
    y = np.repeat(np.arange(3), n_per_cluster)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]
