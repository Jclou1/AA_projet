from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression  # <--- Ajout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


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
