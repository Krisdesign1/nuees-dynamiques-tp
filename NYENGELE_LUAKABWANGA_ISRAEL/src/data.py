"""Génération d'un jeu de données synthétique propre à cette seconde résolution."""
from __future__ import annotations
import numpy as np


def make_demo_data(n_per_cluster: int = 90, random_state: int = 17):
    rng = np.random.default_rng(random_state)

    # Groupe 0 : ellipse inclinée positivement.
    t0 = rng.normal(size=(n_per_cluster, 2))
    A0 = np.array([[1.65, 0.75], [0.15, 0.48]])
    c0 = t0 @ A0.T + np.array([-3.5, 1.5])

    # Groupe 1 : ellipse inclinée négativement.
    t1 = rng.normal(size=(n_per_cluster, 2))
    A1 = np.array([[1.55, -0.70], [0.10, 0.52]])
    c1 = t1 @ A1.T + np.array([3.4, 1.8])

    # Groupe 2 : groupe plus compact, déplacé vers le bas.
    c2 = rng.normal(loc=np.array([0.1, -3.8]), scale=np.array([0.85, 0.55]), size=(n_per_cluster, 2))

    X = np.vstack([c0, c1, c2])
    y = np.repeat(np.arange(3), n_per_cluster)
    order = rng.permutation(len(X))
    return X[order], y[order]
