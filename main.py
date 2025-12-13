from src.data_loader import load_multiple_races, load_race_data
from src.features import prepare_data, split_data
from src.models import train_models, evaluate_models, feature_importance, train_all_models, evaluate_all_models
from src.visualization import plot_model_performance, plot_degradation_panels, plot_actual_strat_vs_predicted_strat
from src.strat import parse_strategy
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt


def main():
    print("F1 Tire Degradation Predictor")

    # Listes des courses à analyser
    # Bahrain est très abrasif (bon pour voir l'usure)
    races_config = [
        (2024, 'Bahrain'),
        (2024, 'Saudi Arabia'),
        (2024, 'Australia'),
        (2024, 'Japan'),
        (2024, 'China'),
        (2024, 'Miami'),
        (2024, 'Imola'),
        (2024, 'Monaco'),
        (2024, 'Canada'),
        (2024, 'Spain'),
        (2024, 'Austria'),
        (2024, 'UK'),
        (2024, 'Hungary'),
        (2024, 'Belgium'),
        (2024, 'Netherlands'),
        (2024, 'Italy'),
        (2024, 'Azerbaijan'),
        (2024, 'Singapore'),
        (2024, 'USA'),
        (2024, 'Mexico'),
        (2024, 'Brazil'),
        (2024, 'Las Vegas'),
        (2024, 'Qatar'),
        (2024, 'Abu Dhabi')
    ]

    # Chargement des données
    df = load_multiple_races(races_config)
    if df.empty:
        print("Aucune donnée chargée. Vérifiez votre connexion internet ou l'API.")
        return
    
    df_race = {}
    df_team = {}
    df_driver = {}
    for team in df['Team'].unique():
        df_team[team] = df[df['Team'] == team]
        for driver in df_team[team]['DriverNumber'].unique():
            df_driver[driver] = df_team[team][df_team[team]['DriverNumber'] == driver]


    # Testing purpose
    # if '1' in df_driver:
    #     for races in df_driver['1']['Circuit'].unique():
    #         print(f"Race: {races}")
    #         print(df_driver['1'][df_driver['1']['Circuit'] == races])


    # Préparation des features
    print("\n Préparation des données")
    X, y, circuit_map = prepare_data(df_driver['1'])  # Exemple pour VER

    # Séparation Entraînement / Test
    X_train, X_test, y_train, y_test = split_data(X, y, group_col="Circuit_ID")

    actual_strat = parse_strategy(X_test, y_test)

    # =============================
    #  COMPARAISON MULTI-MODÈLES
    # =============================
    print("\n--- Comparaison de plusieurs modèles ---")

    models = train_all_models(X_train, y_train)
    results = evaluate_all_models(models, X_test, y_test)

    print("\nRésumé des performances :")
    for model in results['Model'].unique():
        print(f"{model} : \n RMSE = {results.loc[results['Model'] == model, 'RMSE'].values[0]:.4f}, \n"
                f"MAE = {results.loc[results['Model'] == model, 'MAE'].values[0]:.4f}, \n"
                f"MSE = {results.loc[results['Model'] == model, 'MSE'].values[0]:.4f}, \n"
                f"R2 = {results.loc[results['Model'] == model, 'R2'].values[0]:.4f}, \n"
                f"Strat = {results.loc[results['Model'] == model, 'Strat'].values[0]}, \n"
                f"actual = {actual_strat}\n")

    # Visualisation des performances
    plot_actual_strat_vs_predicted_strat(actual_strat, results, circuit_map, model_name="RandomForest")
    

    print("\nTerminé avec succès !")


if __name__ == "__main__":
    main()
