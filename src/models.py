from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from src.strat import parse_strategy
import pandas as pd
import numpy as np


def train_all_models(X_train, y_train):
    """
    Entraîne plusieurs classificateurs.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        # SVC peut être lent sur de gros datasets, attention
        "SVC_RBF": SVC(kernel="rbf", C=1.0)
    }

    for name, model in models.items():
        # print(f"Entraînement de {name}...")
        model.fit(X_train, y_train)

    return models


def evaluate_all_models(models, X_test, y_test):
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        # Reconstruction de la stratégie pour validation
        st = parse_strategy(X_test, pd.Series(y_pred))

        # Métriques de classification
        acc = accuracy_score(y_test, y_pred)
        # Moyenne pondérée pour multiclasse
        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1_Score": f1,
            "Strat": st
        })

    return pd.DataFrame(results)
