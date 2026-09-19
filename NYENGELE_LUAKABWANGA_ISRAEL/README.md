# TP Science des données - Nuées Dynamiques

**Étudiant : NYENGELE LUAKABWANGA ISRAËL**  
**Université de Kinshasa - Faculté des Sciences et Technologies - Master 1 MSI**

Implémentation alternative **from scratch** du cadre des Nuées Dynamiques de Diday (1971). Le moteur n'utilise pas `scikit-learn` pour le clustering.

## Particularités de cette résolution

- standardisation z-score codée directement ;
- initialisation maximin / farthest-first ;
- plusieurs redémarrages (`n_init`) et conservation de la meilleure solution ;
- réparation d'une classe vide par scission du plus grand groupe ;
- cinq représentations : `point`, `points`, `axes`, `distribution`, `structure` ;
- métriques silhouette et ARI codées dans le projet ;
- interface CLI et interface Streamlit.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ligne de commande

Le programme peut fonctionner directement avec le jeu synthétique intégré :

```bash
python app.py --clusters 3 --representation points --n-prototypes 5
```

Pour générer le fichier `donnees.csv` et reproduire les résultats expérimentaux :

```bash
python experiments/run_experiments.py
```

Puis il est aussi possible de lancer :

```bash
python app.py --input donnees.csv --clusters 3 --representation points --n-prototypes 5
```

Autres modes :

```bash
python app.py --clusters 3 --representation point
python app.py --clusters 3 --representation axes --n-axes 1
python app.py --clusters 3 --representation distribution
python app.py --clusters 3 --representation structure
```

## Interface graphique

```bash
python -m streamlit run streamlit_app.py
```

## Tests

```bash
python -m pytest -q
```

## Rapport et références

Le rapport LaTeX est disponible dans `report/rapport.tex`. Les sources scientifiques et la traçabilité des choix sont détaillées dans `REFERENCES.md`.
