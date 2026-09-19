"""Moteur alternatif des Nuées Dynamiques.

La boucle générale « affecter aux représentants puis reconstruire les représentants »
reprend le principe de la méthode des Nuées Dynamiques de Diday (1971). Cette
résolution adopte volontairement une stratégie distincte : standardisation interne,
initialisation maximin, plusieurs redémarrages, réparation d'une classe vide par
scission du plus grand groupe et nouveaux coûts de représentation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

from .preprocessing import Standardizer
from .representations import Representative, make_representative


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


@dataclass
class _RunResult:
    labels: np.ndarray
    representatives: List[Representative]
    history: List[float]
    n_iter: int
    converged: bool


@dataclass
class AlternativeDynamicClouds:
    n_clusters: int = 3
    representation: str = "points"
    n_prototypes: int = 5
    q_neighbors: int = 2
    n_axes: int = 1
    min_variance: float = 1e-3
    standardize: bool = True
    n_init: int = 5
    max_iter: int = 100
    tol: float = 1e-5
    random_state: int | None = 42
    verbose: bool = False

    labels_: np.ndarray | None = None
    representatives_: List[Representative] | None = None
    objective_history_: List[float] | None = None
    n_iter_: int = 0
    converged_: bool = False
    best_run_: int = -1
    scaler_: Standardizer | None = None

    def _validate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X doit être une matrice 2D.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contient NaN ou infini.")
        if self.n_clusters < 2:
            raise ValueError("n_clusters doit être >= 2.")
        if len(X) < self.n_clusters:
            raise ValueError("Le nombre d'observations doit être >= n_clusters.")
        if self.n_init < 1:
            raise ValueError("n_init doit être >= 1.")
        return X

    def _new_rep(self) -> Representative:
        return make_representative(
            self.representation,
            n_prototypes=self.n_prototypes,
            q_neighbors=self.q_neighbors,
            n_axes=self.n_axes,
            min_variance=self.min_variance,
        )

    def _maximin_seed_labels(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(X)
        seeds = [int(rng.integers(0, n))]
        while len(seeds) < self.n_clusters:
            d2 = _sqdist(X, X[np.array(seeds)])
            nearest = np.min(d2, axis=1)
            nearest[seeds] = -np.inf
            seeds.append(int(np.argmax(nearest)))
        return np.argmin(_sqdist(X, X[np.array(seeds)]), axis=1)

    def _fit_representatives(self, X: np.ndarray, labels: np.ndarray) -> List[Representative]:
        reps: List[Representative] = []
        for k in range(self.n_clusters):
            rep = self._new_rep()
            rep.fit(X[labels == k])
            reps.append(rep)
        return reps

    @staticmethod
    def _cost_matrix(X: np.ndarray, reps: List[Representative]) -> np.ndarray:
        return np.column_stack([rep.cost(X) for rep in reps])

    def _repair_empty(self, X: np.ndarray, labels: np.ndarray, reps: List[Representative]) -> np.ndarray:
        labels = labels.copy()
        counts = np.bincount(labels, minlength=self.n_clusters)
        while np.any(counts == 0):
            empty = int(np.flatnonzero(counts == 0)[0])
            donor = int(np.argmax(counts))
            donor_idx = np.flatnonzero(labels == donor)
            if len(donor_idx) <= 1:
                candidates = np.flatnonzero(counts > 1)
                if len(candidates) == 0:
                    raise RuntimeError("Impossible de réparer une classe vide.")
                donor = int(candidates[np.argmax(counts[candidates])])
                donor_idx = np.flatnonzero(labels == donor)
            donor_cost = reps[donor].cost(X[donor_idx])
            moved = int(donor_idx[np.argmax(donor_cost)])
            labels[moved] = empty
            counts[donor] -= 1
            counts[empty] += 1
        return labels

    def _one_run(self, X: np.ndarray, rng: np.random.Generator) -> _RunResult:
        labels = self._maximin_seed_labels(X, rng)
        counts = np.bincount(labels, minlength=self.n_clusters)
        if np.any(counts == 0):
            for empty in np.flatnonzero(counts == 0):
                donor = int(np.argmax(counts))
                idx = np.flatnonzero(labels == donor)[-1]
                labels[idx] = int(empty)
                counts[donor] -= 1
                counts[empty] += 1

        reps = self._fit_representatives(X, labels)
        history: List[float] = []
        converged = False
        previous = None

        for it in range(1, self.max_iter + 1):
            costs = self._cost_matrix(X, reps)
            new_labels = np.argmin(costs, axis=1)
            new_labels = self._repair_empty(X, new_labels, reps)
            new_reps = self._fit_representatives(X, new_labels)
            updated_costs = self._cost_matrix(X, new_reps)
            objective = float(np.sum(updated_costs[np.arange(len(X)), new_labels]))
            history.append(objective)

            stable = np.array_equal(new_labels, labels)
            relative = np.inf if previous is None else abs(previous - objective) / max(abs(previous), 1.0)
            if self.verbose:
                print(f"it={it:03d} objectif={objective:.6f} variation={relative:.3e} stable={stable}")

            labels, reps = new_labels, new_reps
            if stable or (previous is not None and relative <= self.tol):
                converged = True
                return _RunResult(labels, reps, history, it, converged)
            previous = objective

        return _RunResult(labels, reps, history, self.max_iter, converged)

    def fit(self, X: np.ndarray) -> "AlternativeDynamicClouds":
        X0 = self._validate(X)
        if self.standardize:
            self.scaler_ = Standardizer().fit(X0)
            Xt = self.scaler_.transform(X0)
        else:
            self.scaler_ = None
            Xt = X0.copy()

        master = np.random.default_rng(self.random_state)
        best: _RunResult | None = None
        best_obj = np.inf
        best_run = -1

        for run in range(self.n_init):
            run_seed = int(master.integers(0, np.iinfo(np.int32).max))
            result = self._one_run(Xt, np.random.default_rng(run_seed))
            final_obj = result.history[-1]
            if final_obj < best_obj:
                best_obj = final_obj
                best = result
                best_run = run

        assert best is not None
        self.labels_ = best.labels.copy()
        self.representatives_ = best.representatives
        self.objective_history_ = list(best.history)
        self.n_iter_ = best.n_iter
        self.converged_ = best.converged
        self.best_run_ = best_run
        self._X_fit_original_ = X0.copy()
        self._X_fit_transformed_ = Xt.copy()
        return self

    def _transform_for_prediction(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.scaler_.transform(X) if self.scaler_ is not None else X

    def decision_costs(self, X: np.ndarray) -> np.ndarray:
        if self.representatives_ is None:
            raise RuntimeError("fit doit être appelé avant decision_costs.")
        Xt = self._transform_for_prediction(X)
        return self._cost_matrix(Xt, self.representatives_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmin(self.decision_costs(X), axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_.copy()

    def summary(self) -> Dict[str, Any]:
        if self.labels_ is None:
            raise RuntimeError("fit doit être appelé avant summary.")
        return {
            "n_clusters": self.n_clusters,
            "representation": self.representation,
            "standardize": self.standardize,
            "n_init": self.n_init,
            "best_run": self.best_run_,
            "iterations": self.n_iter_,
            "converged": self.converged_,
            "final_objective": self.objective_history_[-1],
            "cluster_sizes": np.bincount(self.labels_, minlength=self.n_clusters).tolist(),
            "representatives": [rep.describe() for rep in self.representatives_],
        }
