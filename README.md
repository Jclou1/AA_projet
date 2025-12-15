# 🏎️ F1 Race Strategy Predictor

**Cours :** Introduction à l'apprentissage automatique (GIF-4101 / GIF-7005)
**Université :** Université Laval
**Session :** Automne 2025

Ce projet vise à appliquer des techniques d'apprentissage automatique supervisé pour modéliser et prédire les **stratégies de course** en Formule 1. L'objectif est de développer un "Assistant Stratège" capable d'anticiper les choix de pneumatiques (_Soft, Medium, Hard_) et les fenêtres d'arrêt aux stands à partir de données historiques.

## 🎯 Contexte et Objectifs

Dans la F1 moderne, la stratégie est aussi déterminante que la performance pure. Ce projet répond à la problématique suivante : **Comment prédire la séquence optimale de pneumatiques pour une course future en se basant sur l'historique ?**

Nous avons développé un pipeline complet qui :

1.  Extrait les données de télémétrie via l'API **FastF1**.
2.  Entraîne des **classificateurs** (Random Forest, Gradient Boosting, etc.) pour prédire le composé de pneu idéal tour par tour.
3.  Reconstitue des stratégies de course cohérentes (séquences de relais) via des algorithmes de lissage.
4.  Analyse l'impact du volume de données sur la performance (Courbe d'apprentissage).

## 📂 Structure du Projet

```text
projet_f1/
├── data/                  # Cache local des données FastF1 (créé automatiquement)
├── outputs/               # Graphiques générés (Stratégies, Accuracy, Features)
├── src/
│   ├── __init__.py
[cite_start]│   ├── data_loader.py     # Extraction et nettoyage (Filtre SC/VSC, Pluie) [cite: 5]
[cite_start]│   ├── features.py        # Ingénierie des features (TyreLife, TrackTemp...) [cite: 4]
[cite_start]│   ├── models.py          # Entraînement et évaluation (RF, LogReg, GBM...) [cite: 6]
[cite_start]│   ├── strat.py           # Reconstruction des stratégies (Parsing & Lissage) [cite: 7]
[cite_start]│   └── visualization.py   # Génération des graphiques d'analyse [cite: 3]
[cite_start]├── main.py                # Script principal d'exécution [cite: 8]
[cite_start]├── gp_finder.py           # Utilitaire pour lister les GP disponibles [cite: 9]
[cite_start]├── requirements.txt       # Dépendances Python [cite: 10]
└── README.md              # Documentation du projet
```

## 🚀 Installation

1.  **Cloner le dépôt :**

    ```bash
    git clone <votre-repo-url>
    cd projet_f1
    ```

2.  **Installer les dépendances :**
    Il est recommandé d'utiliser un environnement virtuel (venv ou conda).

    ```bash
    pip install -r requirements.txt
    ```

    _Principales librairies :_ `fastf1`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`.

## ▶️ Utilisation

Pour lancer l'analyse complète (entraînement, courbes d'apprentissage et génération des graphiques), exécutez simplement le script principal :

```bash
python main.py
```

**Ce que fait le script :**

1.  Charge progressivement les données historiques (ex: 2019, puis 2019-2020, etc.) pour analyser la courbe d'apprentissage.
2.  Entraîne plusieurs modèles sur les pilotes cibles (VER, LEC, HAM, etc.).
3.  Teste la performance sur une course cible (ex: Abu Dhabi 2025).
4.  Génère et sauvegarde les graphiques dans le dossier `outputs/`.

## ⚙️ Méthodologie

### 1\. Prétraitement des Données (`src/data_loader.py`)

- Utilisation de `FastF1` pour récupérer la télémétrie.
- **Filtrage :** Exclusion des sessions sous la pluie et nettoyage des tours non représentatifs (tours de sortie/entrée, Safety Car).

### 2\. Feature Engineering (`src/features.py`)

Transformation des données brutes en variables prédictives :

- **État Course :** `LapNumber`, `TrackStatus` (SC/VSC).
- **Physique Pneu :** `TyreLife` (Âge du train de pneus).
- **Conditions :** `TrackTemp`, `AirTemp`.
- **Contexte :** `Position`, `Team`.

### 3\. Modélisation (`src/models.py`)

Comparaison de plusieurs algorithmes de classification :

- **Random Forest Classifier** (Modèle principal, robuste).
- Logistic Regression (Baseline).
- Gradient Boosting & KNN.

### 4\. Reconstruction de Stratégie (`src/strat.py`)

Conversion des prédictions tour par tour en une stratégie lisible (ex: `SOFT (15 tours) -> HARD (20 tours)`). Implémentation d'une logique de lissage pour éviter les changements de pneus irréalistes sur un seul tour.

## 📊 Résultats et Visualisations

Les résultats sont sauvegardés automatiquement dans le dossier `outputs/`. Les analyses incluent :

1.  **Comparaison Réel vs Prédit :** Graphique montrant la stratégie exécutée par le pilote vs celle prédite par l'IA.
2.  **Courbe d'Apprentissage (Learning Curve) :** Analyse de l'évolution de la précision (Accuracy) en fonction du nombre d'années d'historique incluses.
3.  **Comparaison des Modèles :** Bar chart comparant l'Accuracy et le F1-Score des différents algorithmes.
4.  **Importance des Features :** Classement des variables (ex: `TyreLife`, `LapNumber`) ayant le plus d'impact sur la décision du modèle.
