"""Representations used by the Dynamic Clouds algorithm.

Scientific references
---------------------
- Diday (1971): dynamic clusters / representative elements (étalons).
- MacQueen (1967), Lloyd (1982): centroid / k-means case.
- Gonzalez (1985): farthest-first inspiration for prototype coverage.
- Kaufman & Rousseeuw (1990): medoid/PAM inspiration for local swaps.
- Jolliffe & Cadima (2016): principal component axes from covariance.
- Mahalanobis (1936): covariance-aware quadratic distance.
- Rousseeuw & Croux (1993): MAD and the 1.4826 Gaussian consistency factor.

See ../REFERENCES.md for complete citations. The exact software architecture and
some assignment costs are original implementation choices for this TP.
The clustering logic is implemented without scikit-learn; NumPy is used for
numerical primitives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


class BaseRepresentation:
    """Base interface for a cluster representative."""

    name = "base"

    def fit(self, X: np.ndarray, rng: np.random.Generator) -> "BaseRepresentation":
        raise NotImplementedError

    def cost(self, X: np.ndarray) -> np.ndarray:
        """Return one non-negative assignment cost per observation."""
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"type": self.name}


@dataclass
class PointRepresentation(BaseRepresentation):
    """Single centroid; cf. MacQueen (1967) and Lloyd (1982)."""

    center_: np.ndarray | None = None
    name = "point"

    def fit(self, X: np.ndarray, rng: np.random.Generator) -> "PointRepresentation":
        if len(X) == 0:
            raise ValueError("Cannot fit a point representation on an empty cluster.")
        self.center_ = np.mean(X, axis=0)
        return self

    def cost(self, X: np.ndarray) -> np.ndarray:
        diff = X - self.center_
        return np.sum(diff * diff, axis=1)

    def describe(self) -> Dict[str, Any]:
        return {"type": self.name, "center": self.center_.copy()}


@dataclass
class MultiPointRepresentation(BaseRepresentation):
    """Several observed prototypes for one class.

    Multiple representatives follow Diday's (1971) étalon idea. Initial coverage
    is inspired by Gonzalez (1985); local swaps are a simplified medoid/PAM-like
    refinement inspired by Kaufman & Rousseeuw (1990).
    """

    n_representatives: int = 3
    cloud_distance: str = "min"  # min or mean
    local_refinement_steps: int = 2
    prototypes_: np.ndarray | None = None
    name = "points"

    @staticmethod
    def _pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        aa = np.sum(A * A, axis=1)[:, None]
        bb = np.sum(B * B, axis=1)[None, :]
        d2 = aa + bb - 2.0 * (A @ B.T)
        return np.maximum(d2, 0.0)

    def _coverage_objective(self, X: np.ndarray, medoid_idx: np.ndarray) -> float:
        d2 = self._pairwise_sqdist(X, X[medoid_idx])
        return float(np.sum(np.min(d2, axis=1)))

    def fit(self, X: np.ndarray, rng: np.random.Generator) -> "MultiPointRepresentation":
        if len(X) == 0:
            raise ValueError("Cannot fit a multi-point representation on an empty cluster.")

        m = max(1, min(int(self.n_representatives), len(X)))

        # Greedy farthest-first initialization anchored at the point nearest the mean.
        mean = np.mean(X, axis=0)
        first = int(np.argmin(np.sum((X - mean) ** 2, axis=1)))
        selected = [first]
        while len(selected) < m:
            d2 = self._pairwise_sqdist(X, X[np.array(selected)])
            nearest = np.min(d2, axis=1)
            nearest[selected] = -1.0
            selected.append(int(np.argmax(nearest)))

        medoids = np.array(selected, dtype=int)

        # Lightweight swap refinement (PAM-style) to reduce coverage error.
        if len(X) > m:
            current = self._coverage_objective(X, medoids)
            for _ in range(max(0, self.local_refinement_steps)):
                improved = False
                medoid_set = set(medoids.tolist())
                non_medoids = [i for i in range(len(X)) if i not in medoid_set]
                # Limit candidate count for speed on large clusters.
                if len(non_medoids) > 80:
                    non_medoids = rng.choice(non_medoids, size=80, replace=False).tolist()
                for pos in range(len(medoids)):
                    best_idx = medoids[pos]
                    best_obj = current
                    for cand in non_medoids:
                        proposal = medoids.copy()
                        proposal[pos] = cand
                        obj = self._coverage_objective(X, proposal)
                        if obj + 1e-12 < best_obj:
                            best_obj = obj
                            best_idx = cand
                    if best_idx != medoids[pos]:
                        medoids[pos] = best_idx
                        current = best_obj
                        improved = True
                if not improved:
                    break

        self.prototypes_ = X[medoids].copy()
        return self

    def cost(self, X: np.ndarray) -> np.ndarray:
        d2 = self._pairwise_sqdist(X, self.prototypes_)
        if self.cloud_distance == "min":
            return np.min(d2, axis=1)
        if self.cloud_distance == "mean":
            return np.mean(d2, axis=1)
        raise ValueError("cloud_distance must be 'min' or 'mean'.")

    def describe(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "n_representatives": len(self.prototypes_),
            "prototypes": self.prototypes_.copy(),
            "cloud_distance": self.cloud_distance,
        }


@dataclass
class AxesRepresentation(BaseRepresentation):
    """Centroid plus local principal axes; see Jolliffe & Cadima (2016)."""

    n_axes: int = 1
    center_weight: float = 0.03
    center_: np.ndarray | None = None
    axes_: np.ndarray | None = None
    eigenvalues_: np.ndarray | None = None
    name = "axes"

    def fit(self, X: np.ndarray, rng: np.random.Generator) -> "AxesRepresentation":
        if len(X) == 0:
            raise ValueError("Cannot fit an axes representation on an empty cluster.")
        self.center_ = np.mean(X, axis=0)
        d = X.shape[1]
        centered = X - self.center_

        if len(X) <= 1 or d == 1:
            self.axes_ = np.empty((d, 0))
            self.eigenvalues_ = np.empty((0,))
            return self

        cov = (centered.T @ centered) / max(len(X) - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        q = max(1, min(int(self.n_axes), d - 1))
        self.axes_ = vecs[:, :q]
        self.eigenvalues_ = np.maximum(vals[:q], 0.0)
        return self

    def cost(self, X: np.ndarray) -> np.ndarray:
        Z = X - self.center_
        if self.axes_.shape[1] == 0:
            return np.sum(Z * Z, axis=1)
        proj = Z @ self.axes_
        reconstructed = proj @ self.axes_.T
        residual = Z - reconstructed
        orth = np.sum(residual * residual, axis=1)
        # A small centroid term prevents pathological ties for parallel/intersecting axes.
        along = np.sum(proj * proj, axis=1)
        return orth + self.center_weight * along

    def describe(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "center": self.center_.copy(),
            "axes": self.axes_.copy(),
            "eigenvalues": self.eigenvalues_.copy(),
        }


@dataclass
class GaussianDistributionRepresentation(BaseRepresentation):
    """Gaussian representative; covariance term relates to Mahalanobis (1936)."""

    reg_covar: float = 1e-6
    mean_: np.ndarray | None = None
    covariance_: np.ndarray | None = None
    precision_: np.ndarray | None = None
    logdet_: float = 0.0
    name = "distribution"

    def fit(self, X: np.ndarray, rng: np.random.Generator) -> "GaussianDistributionRepresentation":
        if len(X) == 0:
            raise ValueError("Cannot fit a distribution on an empty cluster.")
        self.mean_ = np.mean(X, axis=0)
        d = X.shape[1]
        centered = X - self.mean_
        if len(X) <= 1:
            cov = np.eye(d)
        else:
            cov = (centered.T @ centered) / len(X)
        cov = cov + self.reg_covar * np.eye(d)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov = cov + max(self.reg_covar, 1e-6) * 10.0 * np.eye(d)
            sign, logdet = np.linalg.slogdet(cov)
        self.covariance_ = cov
        self.precision_ = np.linalg.pinv(cov)
        self.logdet_ = float(logdet)
        return self

    def cost(self, X: np.ndarray) -> np.ndarray:
        Z = X - self.mean_
        mahal = np.einsum("ij,jk,ik->i", Z, self.precision_, Z)
        # Constants common to all clusters are omitted.
        return mahal + self.logdet_

    def describe(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "mean": self.mean_.copy(),
            "covariance": self.covariance_.copy(),
        }


@dataclass
class RobustStructureRepresentation(BaseRepresentation):
    """Robust center + scale example; MAD follows Rousseeuw & Croux (1993)."""

    eps: float = 1e-8
    median_: np.ndarray | None = None
    mad_: np.ndarray | None = None
    name = "structure"

    def fit(self, X: np.ndarray, rng: np.random.Generator) -> "RobustStructureRepresentation":
        if len(X) == 0:
            raise ValueError("Cannot fit a structure on an empty cluster.")
        self.median_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.median_), axis=0)
        # 1.4826 makes MAD comparable to standard deviation for Gaussian data.
        self.mad_ = 1.4826 * mad + self.eps
        return self

    def cost(self, X: np.ndarray) -> np.ndarray:
        z = np.abs((X - self.median_) / self.mad_)
        return np.sum(z, axis=1)

    def describe(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "median": self.median_.copy(),
            "mad": self.mad_.copy(),
        }


def build_representation(
    kind: str,
    *,
    n_representatives: int = 3,
    cloud_distance: str = "min",
    n_axes: int = 1,
    reg_covar: float = 1e-6,
) -> BaseRepresentation:
    kind = kind.lower().strip()
    if kind == "point":
        return PointRepresentation()
    if kind == "points":
        return MultiPointRepresentation(
            n_representatives=n_representatives,
            cloud_distance=cloud_distance,
        )
    if kind == "axes":
        return AxesRepresentation(n_axes=n_axes)
    if kind == "distribution":
        return GaussianDistributionRepresentation(reg_covar=reg_covar)
    if kind == "structure":
        return RobustStructureRepresentation()
    raise ValueError(
        "Unknown representation. Choose among: point, points, axes, distribution, structure."
    )
