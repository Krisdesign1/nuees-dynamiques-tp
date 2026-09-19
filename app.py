"""Command-line application for the Dynamic Clouds TP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from src.dynamic_clouds import DynamicClouds
from src.data import make_demo_dataset, standardize
from src.metrics import silhouette_score, adjusted_rand_index


def load_csv(path: str, delimiter: str = ",", skip_header: int = 1):
    arr = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header)
    if arr.ndim == 1:
        arr = arr[:, None]
    # Drop rows containing missing/non-numeric values.
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    return arr


def main():
    p = argparse.ArgumentParser(
        description="Implémentation from scratch de la méthode des nuées dynamiques."
    )
    p.add_argument("--input", type=str, default=None, help="CSV numérique optionnel")
    p.add_argument("--delimiter", type=str, default=",")
    p.add_argument("--skip-header", type=int, default=1)
    p.add_argument("--clusters", type=int, default=3)
    p.add_argument(
        "--representation",
        choices=["point", "points", "axes", "distribution", "structure"],
        default="points",
    )
    p.add_argument("--n-representatives", type=int, default=5)
    p.add_argument("--cloud-distance", choices=["min", "mean"], default="min")
    p.add_argument("--n-axes", type=int, default=1)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--output", type=str, default="results")
    args = p.parse_args()

    if args.input is None:
        X, y_true = make_demo_dataset(random_state=args.seed)
    else:
        X = load_csv(args.input, args.delimiter, args.skip_header)
        y_true = None

    if not args.no_standardize:
        X_model, _, _ = standardize(X)
    else:
        X_model = X

    model = DynamicClouds(
        n_clusters=args.clusters,
        representation=args.representation,
        n_representatives=args.n_representatives,
        cloud_distance=args.cloud_distance,
        n_axes=args.n_axes,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.seed,
        verbose=True,
    )
    labels = model.fit_predict(X_model)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "labels.csv", labels, fmt="%d", delimiter=",")
    np.savetxt(out / "class_costs.csv", model.decision_costs(X_model), delimiter=",")

    summary = model.summary()
    summary["silhouette"] = silhouette_score(X_model, labels)
    if y_true is not None and len(np.unique(y_true)) == args.clusters:
        summary["adjusted_rand_index"] = adjusted_rand_index(y_true, labels)

    # Convert arrays in representative descriptions to lists for JSON.
    def clean(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(clean(summary), f, indent=2, ensure_ascii=False)

    print("\nRésumé")
    print(json.dumps(clean(summary), indent=2, ensure_ascii=False))
    print(f"\nRésultats enregistrés dans: {out.resolve()}")


if __name__ == "__main__":
    main()
