from src.data_loader import load_multiple_races, load_race_data
from src.features import prepare_data
from src.models import train_all_models, evaluate_all_models
from src.visualization import plot_actual_strat_vs_predicted_strat, plot_accuracy_comparison
from src.strat import parse_strategy


def main():
    print("F1 Tire Degradation Predictor")

    # Listes des courses à analyser
    # Bahrain est très abrasif (bon pour voir l'usure)
    races_config = [
        # (2019, 'Abu Dhabi'),
        # (2020, 'Abu Dhabi'),
        # (2021, 'Abu Dhabi'),
        # (2022, 'Abu Dhabi'),
        # (2023, 'Abu Dhabi'),
        # (2024, 'Abu Dhabi'),
        (2019, 'Qatar'),
        (2020, 'Qatar'),
        (2021, 'Qatar'),
        (2022, 'Qatar'),
        (2023, 'Qatar'),
        (2024, 'Qatar'),
    ]

    drivers = ['VER', 'LEC', 'HAM', 'GAS', 'RUS', 'NOR']

    # Chargement des données
    df = load_multiple_races(races_config)
    if df.empty:
        print("Aucune donnée chargée. Vérifiez votre connexion internet ou l'API.")
        return

    average_accuracy_per_model = {}
    # Préparation des features
    for driver in drivers:
        print(f"\n Préparation des données pour {driver}")
        # Only keep data for the specific driver
        X, y = prepare_data(df[df['Driver'] == driver])

        # Train on all data
        X_train, y_train = X, y

        # load Abu Dhabi 2025 for testing
        df_test = load_race_data(2025, 'Abu Dhabi')
        X_test, y_test = prepare_data(df_test[df_test['Driver'] == driver])

        actual_strat = parse_strategy(X_test, y_test)

        # =============================
        #  COMPARAISON MULTI-MODÈLES
        # =============================
        print("\n--- Comparaison de plusieurs modèles ---")

        models = train_all_models(X_train, y_train)
        results = evaluate_all_models(models, X_test, y_test)

        print("\nRésumé des performances :")

        for model in results['Model'].unique():
            # Compute the average accuracy per model across drivers
            if model not in average_accuracy_per_model:
                average_accuracy_per_model[model] = []

            acc = results.loc[results['Model'] == model, 'Accuracy'].values[0]
            f1 = results.loc[results['Model'] == model, 'F1_Score'].values[0]

            average_accuracy_per_model[model].append(acc)

            print(f"{model} : \n Accuracy = {acc:.2%}, \n"
                  f"F1 Score = {f1:.4f}, \n"
                  f"actual = {actual_strat}\n")

        # Visualisation des performances
        plot_actual_strat_vs_predicted_strat(
            actual_strat, results, driver, model_name="RandomForest")

        print("\nTerminé avec succès !")

    print("\n=== Average Accuracy per Model across Drivers ===")
    for model, accuracies in average_accuracy_per_model.items():
        avg_acc = sum(accuracies) / len(accuracies)
        print(f"{model} : Average Accuracy = {avg_acc:.2%}")

    # Plot the average accuracies per model
    plot_accuracy_comparison(average_accuracy_per_model)


if __name__ == "__main__":
    main()
