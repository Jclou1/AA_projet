from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.strat import parse_strategy
import numpy as np
import pandas as pd


def train_models(X_train, y_train):
    # Modèle simple : Régression Linéaire
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Modèle avancé : Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    return rf_model, linear_model


def evaluate_models(rf_model, linear_model, X_test, y_test):
    # Évaluation du Random Forest
    rf_preds = rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)

    # Évaluation du Linéaire
    lin_preds = linear_model.predict(X_test)
    lin_rmse = np.sqrt(mean_squared_error(y_test, lin_preds))
    lin_mae = mean_absolute_error(y_test, lin_preds)

    print(f"Comparaison des performances (RMSE) :")
    print(f"Régression Linéaire (Base) : {lin_rmse:.4f} sec")
    print(f"Random Forest (Avancé)     : {rf_rmse:.4f} sec")

    return rf_preds, rf_rmse, rf_mae, lin_preds, lin_rmse, lin_mae

def feature_importance(model, feature_names):
    importances = model.feature_importances_
    for name, val in zip(feature_names, importances):
        print(f"{name} : {val:.3f}")


def train_all_models(X_train, y_train):
    """
    Entraîne plusieurs modèles de régression sur les mêmes données.
    Retourne un dict {nom_modèle: instance_entraînée}.
    """

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=42
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR_RBF": SVR(kernel="rbf", C=10.0, epsilon=0.1),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models


def evaluate_all_models(models, X_test, y_test):
    """
    Évalue tous les modèles fournis sur le même X_test / y_test.
    Un DataFrame des résultats.
    """
    results = []

    # Predict
    for name, model in models.items():
        y_pred = model.predict(X_test)

        st = parse_strategy(X_test, pd.Series(y_pred))

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Strat" : st
        })

    return pd.DataFrame(results)
