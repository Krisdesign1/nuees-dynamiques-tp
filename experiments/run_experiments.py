"""Run the five representations on the same synthetic dataset and save figures/results."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.dynamic_clouds import DynamicClouds
from src.data import make_demo_dataset, standardize
from src.metrics import silhouette_score, adjusted_rand_index


def plot_partition(X, labels, title, path, model=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=20, alpha=0.85)
    ax.set_xlabel("Variable 1")
    ax.set_ylabel("Variable 2")
    ax.set_title(title)

    if model is not None and model.representation == "points":
        for rep in model.representatives_:
            P = rep.prototypes_
            ax.scatter(P[:, 0], P[:, 1], marker="X", s=85, edgecolors="black")
    elif model is not None and model.representation == "point":
        C = np.vstack([r.center_ for r in model.representatives_])
        ax.scatter(C[:, 0], C[:, 1], marker="X", s=100, edgecolors="black")
    elif model is not None and model.representation == "axes":
        for rep in model.representatives_:
            if rep.axes_.shape[1] > 0:
                v = rep.axes_[:, 0]
                c = rep.center_
                scale = 2.0
                ax.plot([c[0]-scale*v[0], c[0]+scale*v[0]], [c[1]-scale*v[1], c[1]+scale*v[1]], linewidth=2)
                ax.scatter([c[0]], [c[1]], marker="X", s=80, edgecolors="black")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    out = ROOT / "report" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    X_raw, y = make_demo_dataset(n_per_cluster=120, random_state=7)
    X, _, _ = standardize(X_raw)

    # True-data view in original coordinates.
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(X_raw[:, 0], X_raw[:, 1], c=y, s=20, alpha=0.85)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Jeu de données synthétique - classes génératrices")
    fig.tight_layout()
    fig.savefig(out / "dataset.png", dpi=180)
    plt.close(fig)

    configs = [
        ("point", {"n_clusters": 3}),
        ("points", {"n_clusters": 3, "n_representatives": 7, "cloud_distance": "min"}),
        ("axes", {"n_clusters": 3, "n_axes": 1}),
        ("distribution", {"n_clusters": 3}),
        ("structure", {"n_clusters": 3}),
    ]

    rows = []
    for kind, kwargs in configs:
        model = DynamicClouds(
            representation=kind,
            max_iter=100,
            tol=1e-8,
            random_state=13,
            **kwargs,
        ).fit(X)
        labels = model.labels_
        sil = silhouette_score(X, labels)
        ari = adjusted_rand_index(y, labels)
        rows.append({
            "representation": kind,
            "iterations": model.n_iter_,
            "converged": model.converged_,
            "silhouette": sil,
            "ari": ari,
            "objective": model.objective_history_[-1],
        })
        plot_partition(
            X,
            labels,
            f"Nuées dynamiques - {kind}",
            out / f"partition_{kind}.png",
            model,
        )

        # One convergence plot per representation.
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.plot(np.arange(1, len(model.objective_history_) + 1), model.objective_history_, marker="o")
        ax.set_xlabel("Itération")
        ax.set_ylabel("Critère interne")
        ax.set_title(f"Convergence - {kind}")
        fig.tight_layout()
        fig.savefig(out / f"convergence_{kind}.png", dpi=180)
        plt.close(fig)

    with open(results_dir / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("Results")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
