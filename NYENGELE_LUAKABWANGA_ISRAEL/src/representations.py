"""Représentations interchangeables d'une classe.

Références conceptuelles
------------------------
- Diday (1971) : une classe peut être décrite par plusieurs « étalons ».
- MacQueen (1967), Lloyd (1982) : cas ponctuel par centroïde / k-means.
- Gonzalez (1985) : idée de parcours farthest-first pour couvrir un ensemble.
- Pearson (1901) : axes principaux.
- Bishop (2006) : modèle gaussien et log-vraisemblance.
- Tukey (1977) : médiane, quartiles et IQR.

Les coûts précis ci-dessous et l'API logicielle sont des choix d'implémentation
propres à cette résolution du TP.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


def _squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


class Representative:
    name = "base"

    def fit(self, X: np.ndarray) -> "Representative":
        raise NotImplementedError

    def cost(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"type": self.name}


@dataclass
class CentroidRepresentative(Representative):
    center_: np.ndarray | None = None
    name = "point"

    def fit(self, X: np.ndarray) -> "CentroidRepresentative":
        if len(X) == 0:
            raise ValueError("Classe vide.")
        self.center_ = np.mean(X, axis=0)
        return self

    def cost(self, X: np.ndarray) -> np.ndarray:
        return np.sum((np.asarray(X) - self.center_) ** 2, axis=1)

    def describe(self) -> Dict[str, Any]:
        return {"type": self.name, "center": self.center_.tolist()}


@dataclass
class PrototypeCloudRepresentative(Representative):
    n_prototypes: int = 5
    q_neighbors: int = 2
    prototypes_: np.ndarray | None = None
    name = "points"

    def fit(self, X: np.ndarray) -> "PrototypeCloudRepresentative":
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            raise ValueError("Classe vide.")
        m = min(max(1, self.n_prototypes), len(X))
        if m == len(X):
            self.prototypes_ = X.copy()
            return self

        d2 = _squared_distances(X, X)
        first = int(np.argmin(np.sum(np.sqrt(d2), axis=1)))
        selected = [first]

        while len(selected) < m:
            nearest = np.min(d2[:, selected], axis=1)
            nearest[selected] = -np.inf
            selected.append(int(np.argmax(nearest)))
        self.prototypes_ = X[np.array(selected)].copy()
        return self

    def cost(self, X: np.ndarray) -> np.ndarray:
        d2 = _squared_distances(np.asarray(X), self.prototypes_)
        q = min(max(1, self.q_neighbors), self.prototypes_.shape[0])
        nearest_q = np.partition(d2, q - 1, axis=1)[:, :q]
        return np.mean(nearest_q, axis=1)

    def describe(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "n_prototypes": int(len(self.prototypes_)),
            "q_neighbors": int(min(self.q_neighbors, len(self.prototypes_))),
            "prototypes": self.prototypes_.tolist(),
        }


@dataclass
class FactorialSubspaceRepresentative(Representative):
    n_axes: int = 1
    center_: np.ndarray | None = None
    axes_: np.ndarray | None = None
    eigenvalues_: np.ndarray | None = None
    name = "axes"

    def fit(self, X: np.ndarray) -> "FactorialSubspaceRepresentative":
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            raise ValueError("Classe vide.")
        self.center_ = np.mean(X, axis=0)
        Z = X - self.center_
        p = X.shape[1]
        if len(X) == 1 or p == 1:
            self.axes_ = np.eye(p)[:, : min(self.n_axes, p)]
            self.eigenvalues_ = np.zeros(self.axes_.shape[1])
            return self
        cov = (Z.T @ Z) / max(len(X) - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        r = min(max(1, self.n_axes), p)
        self.eigenvalues_ = vals[order[:r]]
        self.axes_ = vecs[:, order[:r]]
        return self

    def cost(self, X: np.ndarray) -> np.ndarray:
        Z = np.asarray(X, dtype=float) - self.center_
        projection = (Z @ self.axes_) @ self.axes_.T
        residual = Z - projection
        return np.sum(residual * residual, axis=1)

    def describe(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "center": self.center_.tolist(),
            "axes": self.axes_.tolist(),
            "eigenvalues": self.eigenvalues_.tolist(),
        }


@dataclass
class DiagonalGaussianRepresentative(Representative):
    min_variance: float = 1e-3
    mean_: np.ndarray | None = None
    variance_: np.ndarray | None = None
    name = "distribution"

    def fit(self, X: np.ndarray) -> "DiagonalGaussianRepresentative":
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            raise ValueError("Classe vide.")
        self.mean_ = np.mean(X, axis=0)
        var = np.var(X, axis=0, ddof=0)
        self.variance_ = np.maximum(var, self.min_variance)
        return self

    def cost(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        diff2 = (X - self.mean_) ** 2
        return np.sum(
            np.log(2.0 * np.pi * self.variance_) + diff2 / self.variance_, axis=1
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "mean": self.mean_.tolist(),
            "variance": self.variance_.tolist(),
        }


@dataclass
class QuantileStructureRepresentative(Representative):
    min_scale: float = 1e-3
    median_: np.ndarray | None = None
    q1_: np.ndarray | None = None
    q3_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    name = "structure"

    def fit(self, X: np.ndarray) -> "QuantileStructureRepresentative":
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            raise ValueError("Classe vide.")
        self.median_ = np.median(X, axis=0)
        self.q1_ = np.quantile(X, 0.25, axis=0)
        self.q3_ = np.quantile(X, 0.75, axis=0)
        iqr = self.q3_ - self.q1_
        self.scale_ = np.maximum(iqr / 1.349, self.min_scale)
        return self

    def cost(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = np.abs(X - self.median_) / self.scale_
        lower = self.q1_ - 1.5 * (self.q3_ - self.q1_)
        upper = self.q3_ + 1.5 * (self.q3_ - self.q1_)
        outside = np.maximum(lower - X, 0.0) + np.maximum(X - upper, 0.0)
        return np.sum(z, axis=1) + 0.5 * np.sum(outside / self.scale_, axis=1)

    def describe(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "median": self.median_.tolist(),
            "q1": self.q1_.tolist(),
            "q3": self.q3_.tolist(),
        }


def make_representative(
    kind: str,
    *,
    n_prototypes: int = 5,
    q_neighbors: int = 2,
    n_axes: int = 1,
    min_variance: float = 1e-3,
) -> Representative:
    key = kind.lower().strip()
    if key == "point":
        return CentroidRepresentative()
    if key == "points":
        return PrototypeCloudRepresentative(n_prototypes=n_prototypes, q_neighbors=q_neighbors)
    if key == "axes":
        return FactorialSubspaceRepresentative(n_axes=n_axes)
    if key == "distribution":
        return DiagonalGaussianRepresentative(min_variance=min_variance)
    if key == "structure":
        return QuantileStructureRepresentative()
    raise ValueError(
        "representation doit être l'une de : point, points, axes, distribution, structure."
    )
