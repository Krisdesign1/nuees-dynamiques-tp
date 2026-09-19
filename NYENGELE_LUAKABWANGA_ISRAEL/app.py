#!/usr/bin/env python3
"""Interface ligne de commande de l'implémentation alternative des Nuées Dynamiques."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from src import AlternativeDynamicClouds
from src.data import make_demo_data
from src.metrics import silhouette_score


def load_numeric_csv(path: str, delimiter: str = ",") -> np.ndarray:
    """Charge un CSV numérique, avec ou sans ligne d'en-tête."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    arr = np.genfromtxt(p, delimiter=delimiter, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        raise ValueError("Le CSV est vide.")
    if np.isnan(arr[0]).any():
        arr = np.genfromtxt(p, delimiter=delimiter, dtype=float, skip_header=1)
        if arr.ndim == 1:
            arr = arr[:, None]
    if not np.all(np.isfinite(arr)):
        raise ValueError("Le CSV doit contenir uniquement des variables numériques sans valeurs manquantes.")
    return arr


def save_results(out_dir: Path, X: np.ndarray, labels: np.ndarray, model: AlternativeDynamicClouds) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "cluster"])
        writer.writerows((i, int(c)) for i, c in enumerate(labels))
    summary = model.summary()
    summary["silhouette"] = silhouette_score(X, labels)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nuées Dynamiques - résolution alternative from scratch")
    p.add_argument("--input", help="CSV numérique. Sans --input, un jeu synthétique est utilisé.")
    p.add_argument("--delimiter", default=",")
    p.add_argument("--clusters", type=int, default=3)
    p.add_argument("--representation", choices=["point", "points", "axes", "distribution", "structure"], default="points")
    p.add_argument("--n-prototypes", type=int, default=5)
    p.add_argument("--q-neighbors", type=int, default=2)
    p.add_argument("--n-axes", type=int, default=1)
    p.add_argument("--n-init", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--output", default="results_cli")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.input:
        X = load_numeric_csv(args.input, args.delimiter)
    else:
        X, _ = make_demo_data()

    model = AlternativeDynamicClouds(
        n_clusters=args.clusters,
        representation=args.representation,
        n_prototypes=args.n_prototypes,
        q_neighbors=args.q_neighbors,
        n_axes=args.n_axes,
        n_init=args.n_init,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.seed,
        standardize=not args.no_standardize,
        verbose=args.verbose,
    )
    labels = model.fit_predict(X)
    save_results(Path(args.output), X, labels, model)

    info = model.summary()
    print("=== Nuées Dynamiques : résolution alternative ===")
    print(f"représentation : {info['representation']}")
    print(f"tailles         : {info['cluster_sizes']}")
    print(f"itérations      : {info['iterations']}")
    print(f"convergence     : {info['converged']}")
    print(f"objectif final  : {info['final_objective']:.6f}")
    print(f"silhouette      : {silhouette_score(X, labels):.4f}")
    print(f"résultats       : {args.output}/")


if __name__ == "__main__":
    main()
