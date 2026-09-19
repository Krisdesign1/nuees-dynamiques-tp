"""Prétraitement numérique écrit sans scikit-learn.

La standardisation z-score est une transformation classique : chaque variable est
centrée par sa moyenne et divisée par son écart-type. Ici elle est codée directement
avec NumPy afin de garder le coeur du TP « from scratch ».
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        scale = np.std(X, axis=0, ddof=0)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer.fit doit être appelé avant transform.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer.fit doit être appelé avant inverse_transform.")
        return np.asarray(X, dtype=float) * self.scale_ + self.mean_
