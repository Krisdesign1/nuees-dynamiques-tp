from pathlib import Path
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import AlternativeDynamicClouds
from src.data import make_demo_data
from src.metrics import silhouette_score, adjusted_rand_index

RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

X, y = make_demo_data(n_per_cluster=90, random_state=17)
np.savetxt(ROOT / "donnees.csv", X, delimiter=",", header="x1,x2", comments="")

fig, ax = plt.subplots(figsize=(6.2,4.8))
ax.scatter(X[:,0], X[:,1], c=y, s=20, alpha=.75)
ax.set_title("Jeu synthétique de validation")
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(alpha=.18)
fig.tight_layout(); fig.savefig(FIG / "dataset.png", dpi=180); plt.close(fig)

rows=[]
models={}
for rep in ["point","points","axes","distribution","structure"]:
    model=AlternativeDynamicClouds(n_clusters=3, representation=rep, n_prototypes=5,
        q_neighbors=2, n_axes=1, n_init=5, max_iter=100, random_state=42).fit(X)
    models[rep]=model
    rows.append({
        "representation":rep,
        "iterations":model.n_iter_,
        "objective":model.objective_history_[-1],
        "silhouette":silhouette_score(X, model.labels_),
        "ari":adjusted_rand_index(y, model.labels_),
        "sizes":"-".join(map(str,np.bincount(model.labels_, minlength=3)))
    })

with (RES/"benchmark.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

for rep in ["points","axes","distribution","structure"]:
    model=models[rep]
    fig, ax = plt.subplots(figsize=(6.2,4.8))
    ax.scatter(X[:,0], X[:,1], c=model.labels_, s=20, alpha=.75)
    ax.set_title(f"Partition - {rep}")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(alpha=.18)
    fig.tight_layout(); fig.savefig(FIG/f"clusters_{rep}.png", dpi=180); plt.close(fig)

print(rows)
