# TP Science des données - Méthode des Nuées Dynamiques

Implémentation **from scratch** inspirée de la méthode des nuées dynamiques d'Edwin Diday (1971), sans bibliothèque de clustering (pas de scikit-learn pour l'algorithme).

## Représentations disponibles

L'utilisateur choisit la façon de représenter chaque classe :

- `point` : un centroïde unique -> cas k-means ;
- `points` : plusieurs points représentatifs (prototypes) ;
- `axes` : centroïde + axes factoriels principaux ;
- `distribution` : distribution gaussienne (moyenne + covariance) ;
- `structure` : exemple de structure robuste (médiane + MAD), extensible.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
```

## Exécution rapide

Jeu synthétique inclus :

```bash
python app.py --representation points --clusters 3 --n-representatives 7
```

Cas k-means :

```bash
python app.py --representation point --clusters 3
```

Axes factoriels :

```bash
python app.py --representation axes --clusters 3 --n-axes 1
```

Distribution :

```bash
python app.py --representation distribution --clusters 3
```

CSV numérique d’exemple inclus (`donnees.csv`, 240 observations synthétiques, 4 groupes générés pour le TP) :

```bash
python app.py --input donnees.csv --clusters 4 --representation points --n-representatives 5
```

Le chargeur ignore par défaut la première ligne du CSV (`--skip-header 1`). Le fichier fourni possède donc l’en-tête `x,y`.

## Interface interactive

```bash
streamlit run streamlit_app.py
```

L'interface permet de charger un CSV et de choisir la représentation de la nuée avant de lancer la classification.

## Expériences et figures du rapport

```bash
python experiments/run_experiments.py
```

Les figures sont générées dans `report/figures/` et les métriques dans `results/benchmark.csv`.

## Tests

```bash
pytest -q
```

## Structure

```text
.
├── app.py
├── streamlit_app.py
├── src/
│   ├── dynamic_clouds.py
│   ├── representations.py
│   ├── metrics.py
│   └── data.py
├── experiments/run_experiments.py
├── tests/test_dynamic_clouds.py
├── report/rapport.tex
└── requirements.txt
```

## Traçabilité scientifique

Le rapport distingue explicitement :

- les éléments repris de la méthode historique de Diday ;
- les méthodes classiques utilisées comme briques scientifiques (k-means, ACP, Mahalanobis, MAD, silhouette, ARI, k-means++) ;
- les choix propres au TP (architecture logicielle, fonction de coût des axes, réparation des classes vides, protocole synthétique).

Les références complètes sont rassemblées dans [`REFERENCES.md`](REFERENCES.md) et citées dans `report/rapport.tex`. Les commentaires des modules principaux indiquent également les sources algorithmiques utilisées.

## Référence principale

E. Diday, « Une nouvelle méthode en classification automatique et reconnaissance des formes : la méthode des nuées dynamiques », *Revue de Statistique Appliquée*, vol. 19, n°2, 1971, p. 19-33.
