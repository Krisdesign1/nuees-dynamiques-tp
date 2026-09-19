"""Dynamic Clouds clustering engine.

The assignment/update cycle is inspired by Diday (1971). The seeding routine is
inspired by k-means++ (Arthur & Vassilvitskii, 2007). Empty-cluster repair and
the generic representation interface are project-specific implementation choices.
See ../REFERENCES.md for complete citations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

from .representations import BaseRepresentation, build_representation


@dataclass
class DynamicClouds:
    n_clusters: int = 3
    representation: str = "points"
    n_representatives: int = 3
    cloud_distance: str = "min"
    n_axes: int = 1
    reg_covar: float = 1e-6
    max_iter: int = 100
    tol: float = 1e-6
    random_state: int | None = 42
    verbose: bool = False

    labels_: np.ndarray | None = None
    representatives_: List[BaseRepresentation] | None = None
    objective_history_: List[float] | None = None
    n_iter_: int = 0
    converged_: bool = False

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if len(X) < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values. Clean/impute the data first.")
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be >= 2.")
        return X

    def _new_representation(self) -> BaseRepresentation:
        return build_representation(
            self.representation,
            n_representatives=self.n_representatives,
            cloud_distance=self.cloud_distance,
            n_axes=self.n_axes,
            reg_covar=self.reg_covar,
        )

    def _initialize_labels(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # k-means++-inspired seeding; see Arthur & Vassilvitskii (2007).
        n = len(X)
        centers = [int(rng.integers(0, n))]
        for _ in range(1, self.n_clusters):
            chosen = X[np.array(centers)]
            aa = np.sum(X * X, axis=1)[:, None]
            bb = np.sum(chosen * chosen, axis=1)[None, :]
            d2 = np.maximum(aa + bb - 2.0 * (X @ chosen.T), 0.0)
            nearest = np.min(d2, axis=1)
            nearest[np.array(centers)] = 0.0
            total = float(np.sum(nearest))
            if total <= 0:
                available = np.setdiff1d(np.arange(n), np.array(centers), assume_unique=False)
                centers.append(int(rng.choice(available)))
            else:
                probs = nearest / total
                centers.append(int(rng.choice(n, p=probs)))
        seeds = X[np.array(centers)]
        aa = np.sum(X * X, axis=1)[:, None]
        bb = np.sum(seeds * seeds, axis=1)[None, :]
        d2 = np.maximum(aa + bb - 2.0 * (X @ seeds.T), 0.0)
        labels = np.argmin(d2, axis=1)
        return self._repair_empty_clusters(X, labels, d2)

    def _repair_empty_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        costs: np.ndarray | None = None,
    ) -> np.ndarray:
        labels = labels.copy()
        counts = np.bincount(labels, minlength=self.n_clusters)
        if np.all(counts > 0):
            return labels

        if costs is None:
            # Use global center distance as a simple fallback.
            center = np.mean(X, axis=0)
            hardness = np.sum((X - center) ** 2, axis=1)
        else:
            hardness = np.min(costs, axis=1)

        order = np.argsort(hardness)[::-1]
        used = set()
        for k in np.where(counts == 0)[0]:
            for idx in order:
                old = labels[idx]
                if idx in used:
                    continue
                if counts[old] > 1:
                    counts[old] -= 1
                    labels[idx] = k
                    counts[k] += 1
                    used.add(int(idx))
                    break
        return labels

    def _fit_representatives(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
    ) -> List[BaseRepresentation]:
        reps: List[BaseRepresentation] = []
        for k in range(self.n_clusters):
            Xk = X[labels == k]
            rep = self._new_representation()
            rep.fit(Xk, rng)
            reps.append(rep)
        return reps

    @staticmethod
    def _cost_matrix(X: np.ndarray, reps: List[BaseRepresentation]) -> np.ndarray:
        return np.column_stack([rep.cost(X) for rep in reps])

    def fit(self, X: np.ndarray) -> "DynamicClouds":
        X = self._check_X(X)
        rng = np.random.default_rng(self.random_state)
        labels = self._initialize_labels(X, rng)
        reps = self._fit_representatives(X, labels, rng)

        self.objective_history_ = []
        previous_obj = None
        self.converged_ = False

        for it in range(1, self.max_iter + 1):
            costs = self._cost_matrix(X, reps)
            new_labels = np.argmin(costs, axis=1)
            new_labels = self._repair_empty_clusters(X, new_labels, costs)
            objective = float(np.sum(costs[np.arange(len(X)), new_labels]))
            self.objective_history_.append(objective)

            labels_unchanged = np.array_equal(new_labels, labels)
            rel_change = np.inf
            if previous_obj is not None:
                rel_change = abs(previous_obj - objective) / max(abs(previous_obj), 1.0)

            if self.verbose:
                print(
                    f"iter={it:03d} objective={objective:.6f} "
                    f"rel_change={rel_change:.3e} stable_labels={labels_unchanged}"
                )

            labels = new_labels
            reps = self._fit_representatives(X, labels, rng)

            if labels_unchanged or (previous_obj is not None and rel_change <= self.tol):
                self.converged_ = True
                self.n_iter_ = it
                break

            previous_obj = objective
            self.n_iter_ = it

        self.labels_ = labels
        self.representatives_ = reps
        self._X_fit_ = X.copy()
        return self

    def decision_costs(self, X: np.ndarray) -> np.ndarray:
        """Return the assignment-cost matrix (samples x clusters)."""
        if self.representatives_ is None:
            raise RuntimeError("Call fit before decision_costs.")
        X = np.asarray(X, dtype=float)
        return self._cost_matrix(X, self.representatives_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        costs = self.decision_costs(X)
        return np.argmin(costs, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_.copy()

    def summary(self) -> Dict[str, Any]:
        if self.labels_ is None:
            raise RuntimeError("Call fit before summary.")
        costs = self.decision_costs(self._X_fit_)
        per_cluster_mean_cost = []
        for k in range(self.n_clusters):
            mask = self.labels_ == k
            per_cluster_mean_cost.append(float(np.mean(costs[mask, k])))
        return {
            "n_clusters": self.n_clusters,
            "representation": self.representation,
            "iterations": self.n_iter_,
            "converged": self.converged_,
            "final_objective": self.objective_history_[-1] if self.objective_history_ else None,
            "cluster_sizes": np.bincount(self.labels_, minlength=self.n_clusters).tolist(),
            "per_cluster_mean_cost": per_cluster_mean_cost,
            "representatives": [r.describe() for r in self.representatives_],
        }
