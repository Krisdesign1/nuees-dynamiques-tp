"""Small interactive interface for choosing the class representation."""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.dynamic_clouds import DynamicClouds
from src.data import make_demo_dataset, standardize
from src.metrics import silhouette_score


st.set_page_config(page_title="Nuées dynamiques", layout="wide")
st.title("Méthode des Nuées Dynamiques - implémentation from scratch")
st.write(
    "Choisissez la représentation de chaque classe : point (k-means), plusieurs points, "
    "axes factoriels, distribution gaussienne ou structure robuste."
)

uploaded = st.file_uploader("Importer un CSV numérique (optionnel)", type=["csv"])

with st.sidebar:
    representation = st.selectbox(
        "Représentation de la nuée",
        ["point", "points", "axes", "distribution", "structure"],
        format_func=lambda x: {
            "point": "Point / centroïde (k-means)",
            "points": "Ensemble de points représentatifs",
            "axes": "Axes factoriels",
            "distribution": "Distribution de probabilités",
            "structure": "Structure représentative robuste",
        }[x],
    )
    k = st.slider("Nombre de classes K", 2, 10, 3)
    m = st.slider("Nombre de points représentatifs", 1, 20, 5, disabled=representation != "points")
    q = st.slider("Nombre d'axes", 1, 5, 1, disabled=representation != "axes")
    cloud_distance = st.selectbox("Distance point-nuée", ["min", "mean"], disabled=representation != "points")
    max_iter = st.slider("Itérations max", 10, 300, 100)
    seed = st.number_input("Graine aléatoire", value=42, step=1)
    normalize = st.checkbox("Standardiser les variables", value=True)

if uploaded is None:
    X, _ = make_demo_dataset(random_state=int(seed))
    df = pd.DataFrame(X, columns=["x1", "x2"])
    st.info("Aucun fichier chargé : jeu de données synthétique utilisé.")
else:
    df = pd.read_csv(uploaded)
    numeric = df.select_dtypes(include=[np.number]).dropna()
    if numeric.shape[1] == 0:
        st.error("Le fichier ne contient aucune colonne numérique exploitable.")
        st.stop()
    df = numeric
    X = df.to_numpy(dtype=float)

if normalize:
    X_model, _, _ = standardize(X)
else:
    X_model = X

if st.button("Lancer la classification", type="primary"):
    model = DynamicClouds(
        n_clusters=k,
        representation=representation,
        n_representatives=m,
        cloud_distance=cloud_distance,
        n_axes=q,
        max_iter=max_iter,
        random_state=int(seed),
    ).fit(X_model)

    labels = model.labels_
    sil = silhouette_score(X_model, labels)
    c1, c2, c3 = st.columns(3)
    c1.metric("Itérations", model.n_iter_)
    c2.metric("Convergence", "Oui" if model.converged_ else "Non")
    c3.metric("Silhouette", f"{sil:.3f}")

    result_df = df.copy()
    result_df["cluster"] = labels
    st.dataframe(result_df.head(50), use_container_width=True)

    if X.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=25)
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        ax.set_title(f"Partition obtenue - représentation: {representation}")
        st.pyplot(fig)
    else:
        st.write("Visualisation 2D indisponible : le jeu de données a moins de deux variables.")

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger les résultats CSV", csv, "clusters.csv", "text/csv")
