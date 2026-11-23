# BrainScan

Classification de tumeurs cérébrales à partir d'IRM (4 classes : `glioma`,
`meningioma`, `notumor`, `pituitary`) avec un CNN Keras. Le code du notebook
`notebooks/01_data_loading.ipynb` a été transformé en **modules Python réutilisables**
dans `src/`, avec une interface **Streamlit** pour tester les prédictions.

> **Important** : l'entraînement (`model.fit`) n'est **jamais** lancé par défaut.
> Le programme réutilise `models/best_model.keras` s'il existe. Pour réentraîner,
> il faut explicitement passer `--train True`.

## Structure du projet

```text
BrainScan/
├── app/
│   └── streamlit_app.py        # Interface web (upload + prédiction, sans training)
├── data/                       # Images d'IRM classées par dossier
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── docs/
│   ├── JIRA_PLAN.md            # Epics, tickets, sprints, critères d'acceptation
│   └── jira_tickets.csv        # Import Jira (CSV)
├── models/
│   ├── best_model.keras        # Modèle déjà entraîné (réutilisé par défaut)
│   ├── best_model_classes.json # Ordre des classes (écrit lors du training)
│   ├── training_history.json   # Métriques du dernier entraînement
│   └── evaluation.json         # Résultats d'évaluation
├── notebooks/
│   └── 01_data_loading.ipynb   # Notebook d'origine (référence)
├── src/
│   ├── config/config.py        # Tous les chemins et hyperparamètres
│   ├── data.py                 # Découverte, split, pipeline tf.data
│   ├── model.py                # Architecture, chargement, sauvegarde
│   ├── train.py                # train_model() — seul endroit qui appelle fit()
│   ├── evaluate.py             # Métriques + matrice de confusion
│   ├── predict.py              # predict_image(path)
│   └── main.py                 # CLI : `python -m src.main ...`
├── tests/
│   └── test_pipeline.py        # Tests rapides (aucun appel à fit())
└── pyproject.toml              # Dépendances uv
```

## Installation

```bash
uv sync --dev
```

## Utilisation

### 1. État du modèle (aucun entraînement)

```bash
uv run python -m src.main
uv run python -m src.main --list-classes
uv run python -m src.main --summary
```

### 2. Tester une image

```bash
uv run python -m src.main --predict data/glioma/Te-gl_0010.jpg
```

En Python :

```python
from src.predict import predict_image

result = predict_image("data/glioma/Te-gl_0010.jpg")
print(result.label, result.confidence)
print(result.probabilities)
print(result.as_dict())  # version JSON-friendly
```

`predict_image` accepte aussi un fichier uploadé (objet avec `.read()`), ce que
utilise Streamlit. Le modèle est chargé une seule fois et mis en cache.

### 3. Évaluer le modèle sauvegardé

```bash
uv run python -m src.main --evaluate
uv run python -m src.main --evaluate --max-eval-samples 64 --no-plot
```

Résultats : `models/evaluation.json` et `models/confusion_matrix.png`.

### 4. Réentraîner (optionnel, explicite)

```bash
uv run python -m src.main --train True
uv run python -m src.main --train True --epochs 5 --batch-size 32
```

`--train False` (ou l'absence du flag) ne fait **aucun** entraînement. Quand
`--train True` est utilisé, le meilleur epoch (selon `val_accuracy`) remplace le
modèle sauvegardé, puis les classes et l'historique sont enregistrés à côté.
Le notebook monitorait `val_recall`, une métrique inexistante : c'est corrigé.

### 5. Interface Streamlit

```bash
uv run streamlit run app/streamlit_app.py
```

Uploader une image ou choisir un exemple dans `data/`, puis cliquer sur
**Predict**. L'interface affiche la classe, la confiance et les probabilités
par classe. Elle n'inclut aucun bouton d'entraînement.

### 6. Tests

```bash
uv run pytest -q
```

Les tests vérifient le chargement des données, la forme des batchs, l'ordre des
classes, la CLI et une vraie inférence avec le modèle sauvegardé. Ils n'appellent
jamais `model.fit()`.

## Notes de reproductibilité

- Split stratifié 70 % / 15 % / 15 % avec `SEED = 42` (identique au notebook).
- Images décodées en RGB, redimensionnées en 224×224, pixels **float32 0–255**
  (pas de division par 255) : c'est l'échelle utilisée pour entraîner le modèle
  existant, donc l'inférence doit faire exactement la même chose.
- L'ordre des classes est alphabétique (`glioma`, `meningioma`, `notumor`,
  `pituitary`). Il est sauvegardé dans `models/best_model_classes.json` lors d'un
  réentraînement ; les anciens modèles utilisent l'ordre des dossiers de `data/`.
- Formats supportés : `.keras` (format natif Keras, recommandé) et `.h5` (legacy).
  Le modèle fourni est au format `.keras` ; il n'y a pas de modèle PyTorch `.pt`.

## Plan Jira

Les Epics et tickets sont documentés dans [`docs/JIRA_PLAN.md`](docs/JIRA_PLAN.md)
et [`docs/jira_tickets.csv`](docs/jira_tickets.csv) (5 Epics, 17 tickets, 5 sprints).
Les issues n'ont pas été créées dans Jira : cet environnement n'a pas d'accès
Jira. Importez le CSV via *Project settings → Import issues from CSV*.

## Avertissement

Projet de démonstration/recherche. Ce modèle n'est pas un dispositif médical et
ne doit pas être utilisé pour un diagnostic.
