# Projet AA - Analyse des données des pneus de Formule 1 2024

# 🏎️ Prédicteur de dégradation des pneus en F1

**Cours :** Introduction à l'apprentissage automatique (GIF-4101 / GIF-7005)

**Université :** Université Laval

Ce projet vise à appliquer des techniques d'apprentissage automatique aux données de télémétrie de Formule 1 afin de modéliser et prédire la **dégradation des pneus**. L'objectif est de fournir un outil d'aide à la décision stratégique capable d'identifier le moment optimal pour un arrêt aux stands.

## 📋 Table des matières

1.  [Contexte et Problématique](#🧐-contexte-et-problématique)
2.  [Objectifs du projet](#🎯-objectifs-du-projet)
3.  [Source des données](#💾-source-des-données)
4.  [Méthodologie et Pipeline ML](#⚙️-méthodologie-et-pipeline-ml)
5.  [Installation et Utilisation](#🚀-installation-et-utilisation)

## 🧐 Contexte et Problématique

Dans la Formule 1 moderne, les écuries génèrent des quantités massives de données télémétriques. Cependant, ces données brutes sont complexes, peu documentées et difficiles à corréler.

L'un des enjeux stratégiques majeurs est la gestion des pneus. Contrairement au niveau de carburant, l'usure des pneus n'est pas mesurable directement par un capteur. Elle doit être inférée à partir de la performance en piste.

**Le défi de Machine Learning :**
Isoler l'effet de la dégradation des pneus sur le temps au tour parmi de multiples facteurs confondants :

- **Masse de carburant :** La voiture s'allège à un taux \~1.7kg/tour, ce qui améliore naturellement les temps au tour.
- **Gomme :** Les pneus tendres (Soft) sont rapides mais s'usent vite; les durs (Hard) sont lents mais durables.
- **Conditions de piste :** Évolution de la température et du dépôt de gomme ("track evolution").

Notre modèle cherche à dissocier ces variables pour prédire le "cliff" (chute brutale de performance) des pneus.

## 🎯 Objectifs du projet

Le projet se concentre sur trois axes principaux :

1.  **Ingénierie des données :** Transformer les flux bruts de l'API FastF1 en un jeu de données structuré pour le ML (nettoyage des tours sous Safety Car, filtrage des erreurs de pilotage).
2.  **Modélisation prédictive :** Entraîner un modèle de régression (Random Forest / MLP) pour estimer le temps au tour attendu ($Y$) en fonction de l'âge du pneu, du composé et du contexte de course ($X$).
3.  **Visualisation stratégique :** Générer des courbes de dégradation comparatives (ex: Soft vs Hard) pour visualiser les points de croisement stratégiques.

## 💾 Source des données

Les données proviennent de la librairie open-source **FastF1**.

- **Origine :** Flux de télémétrie officiels de la F1 (Live Timing).
- **Fiabilité :** Données maintenues par la communauté, couvrant les saisons 2018 à aujourd'hui.
- **Contenu :** Télémétrie par tour, météo, type de pneus, position GPS.

## ⚙️ Méthodologie et Pipeline ML

Nous utilisons Python et l'écosystème Scikit-Learn/Pandas. Notre pipeline suit les étapes suivantes :

### 1\. Collecte et Nettoyage (`src/data_loader.py`)

- Extraction des sessions de course via l'API.
- **Filtrage agressif :** Suppression des tours non représentatifs (tours de sortie/entrée des stands, drapeaux jaunes, Safety Car, pluie).
- Seuls les tours "lancés" (Flying Laps) sont conservés.

### 2\. Feature Engineering (`src/features.py`)

Création des variables explicatives pour le modèle :

- `TyreLife` : Âge du pneu en tours.
- `Compound` : Encodage (One-Hot ou Ordinal) du type de gomme (Soft/Medium/Hard).
- `FuelProxy` : Utilisation du numéro de tour (`LapNumber`) comme proxy inversé de la charge carburant.
- `TrackTemp` : Température de la piste (impacte la dégradation thermique).

### 3\. Modélisation (`src/models.py`)

Nous comparons plusieurs approches pour capturer la non-linéarité de l'usure :

- **Baseline :** Régression Linéaire.
- **Modèle principal :** Random Forest Regressor (capable de capturer les seuils de dégradation non-linéaires).

### 4\. Évaluation

- Métrique principale : RMSE (Root Mean Square Error) sur le temps au tour.
- Validation croisée sur des Grands Prix non vus lors de l'entraînement pour tester la généralisation.

## 🚀 Installation et Utilisation

1.  **Cloner le dépôt :**

    ```bash
    git clone git@github.com:Jclou1/AA_projet.git
    cd AA_projet
    ```

2.  **Installer les dépendances :**
    Il est recommandé d'utiliser un environnement virtuel.

    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows : venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Lancer l'analyse :**

    ```bash
    python main.py
    ```

    _Le script téléchargera automatiquement les données nécessaires via FastF1 (mise en cache automatique)._
