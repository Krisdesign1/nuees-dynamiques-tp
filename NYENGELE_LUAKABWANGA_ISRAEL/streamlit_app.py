"""Interface Streamlit de la seconde résolution du TP."""
from __future__ import annotations
import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src import AlternativeDynamicClouds
from src.data import make_demo_data
from src.metrics import silhouette_score


st.set_page_config(page_title="Nuées Dynamiques - TP", layout="wide")
st.title("Implémentation des Nuées Dynamiques - résolution alternative")
st.caption("NYENGELE LUAKABWANGA ISRAËL · Master 1 MSI · Science des données")

with st.sidebar:
    st.header("Paramètres")
    uploaded = st.file_uploader("Jeu de données CSV numérique", type=["csv"])
    k = st.slider("Nombre de classes K", 2, 8, 3)
    label_to_key = {
        "Point / centroïde": "point",
        "Nuée de prototypes": "points",
        "Axes factoriels": "axes",
        "Distribution gaussienne diagonale": "distribution",
        "Structure robuste par quantiles": "structure",
    }
    choice = st.selectbox("Représentation", list(label_to_key))
    rep = label_to_key[choice]
    n_prototypes = st.slider("Nombre de prototypes", 1, 15, 5, disabled=rep != "points")
    q_neighbors = st.slider("Prototypes proches agrégés (q)", 1, 5, 2, disabled=rep != "points")
    n_axes = st.slider("Nombre d'axes", 1, 3, 1, disabled=rep != "axes")
    n_init = st.slider("Redémarrages", 1, 15, 5)
    standardize = st.checkbox("Standardiser les variables", value=True)
    seed = st.number_input("Graine aléatoire", value=42, step=1)

if uploaded is None:
    X, y_true = make_demo_data()
    st.info("Aucun CSV chargé : utilisation du jeu synthétique fourni avec le projet.")
else:
    raw = uploaded.getvalue()
    X = np.genfromtxt(io.BytesIO(raw), delimiter=",", dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if np.isnan(X[0]).any():
        X = np.genfromtxt(io.BytesIO(raw), delimiter=",", dtype=float, skip_header=1)
        if X.ndim == 1:
            X = X[:, None]
    y_true = None
    if not np.all(np.isfinite(X)):
        st.error("Le CSV doit être numérique et sans valeur manquante.")
        st.stop()

if X.shape[1] < 2:
    st.error("L'interface graphique demande au moins deux variables numériques.")
    st.stop()

model = AlternativeDynamicClouds(
    n_clusters=k,
    representation=rep,
    n_prototypes=n_prototypes,
    q_neighbors=q_neighbors,
    n_axes=n_axes,
    n_init=n_init,
    standardize=standardize,
    random_state=int(seed),
)

try:
    labels = model.fit_predict(X)
except Exception as exc:
    st.exception(exc)
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Observations", len(X))
m2.metric("Variables", X.shape[1])
m3.metric("Itérations", model.n_iter_)
m4.metric("Silhouette", f"{silhouette_score(X, labels):.3f}")

left, right = st.columns([1.4, 1])
with left:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=30, alpha=0.78)
    ax.set_xlabel("Variable 1")
    ax.set_ylabel("Variable 2")
    ax.set_title("Partition obtenue")
    ax.grid(alpha=0.18)
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Résumé")
    st.json(model.summary(), expanded=False)
    st.subheader("Évolution de l'objectif")
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
    ax2.plot(np.arange(1, len(model.objective_history_) + 1), model.objective_history_, marker="o")
    ax2.set_xlabel("Itération")
    ax2.set_ylabel("Objectif")
    ax2.grid(alpha=0.18)
    st.pyplot(fig2, clear_figure=True)

st.markdown("""
### Principe de cette résolution
Le moteur conserve l'alternance générale de Diday (1971), mais utilise une stratégie
indépendante : standardisation interne, initialisation maximin, plusieurs redémarrages,
réparation des classes vides par scission du plus grand groupe et représentations
spécifiques au mode choisi. Les références complètes figurent dans `REFERENCES.md`.
""")
