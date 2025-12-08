from src.data_loader import load_multiple_races
from src.features import prepare_data, split_data
from src.models import train_models, evaluate_models
from src.visualization import plot_predictions, plot_degradation_curve


def main():
    print("F1 Tire Degradation Predictor")

    # Listes des courses à analyser
    # Bahrain est très abrasif (bon pour voir l'usure)
    races_config = [
        (2024, 'Bahrain'),
        (2023, 'Bahrain'),
        (2022, 'Bahrain'),
        (2024, 'Saudi Arabia'),
        (2023, 'Saudi Arabia'),
        (2022, 'Saudi Arabia'),
        (2024, 'Australia'),
        (2023, 'Australia'),
        (2022, 'Italy')
    ]

    # Chargement des données
    df = load_multiple_races(races_config)

    if df.empty:
        print("Aucune donnée chargée. Vérifiez votre connexion internet ou l'API.")
        return

    # Préparation des features
    print("\n Préparation des données")
    X, y, le_circuit = prepare_data(df)

    # Séparation Entraînement / Test
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Données d'entraînement : {X_train.shape[0]} tours")
    print(f"Données de test : {X_test.shape[0]} tours")

    # Entraînement
    print("\nEntraînement du modèle")
    rf_model, linear_model = train_models(X_train, y_train)

    # Évaluation
    print("\n Évaluation")
    rf_preds, rf_rmse, rf_mae, lin_preds, lin_rmse, lin_mae = evaluate_models(
        rf_model, linear_model, X_test, y_test)

    # Visualisation
    print("\n Génération des graphiques")
    # Graphique 1 : Précision du modèle
    plot_predictions(y_test, rf_preds,
                     title="Réalité vs Prédictions (Random Forest)")
    plot_predictions(y_test, lin_preds,
                     title="Réalité vs Prédictions (Linéaire)")

    # Graphique 2 : Simulation Stratégique
    # On prend l'ID du circuit de Bahrain
    try:
        bahrain_id = le_circuit.transform(['Bahrain'])[0]
        plot_degradation_curve(rf_model, circuit_id=bahrain_id, track_temp=35,
                               title="Courbe de Dégradation des Pneus (Random Forest)")
        plot_degradation_curve(
            linear_model, circuit_id=bahrain_id, track_temp=35, title="Courbe de Dégradation des Pneus (Linéaire)")
    except:
        print("Circuit Bahrain non trouvé pour la simulation.")

    # Log la performance finale du modèle dans un fichier texte
    with open("outputs/model_performance.txt", "a") as f:
        f.write("Performance du modèle Random Forest Regressor\n")
        f.write(f"RMSE: {rf_rmse:.4f} secondes\n")
        f.write(f"MAE: {rf_mae:.4f} secondes\n")
        f.write("\n")
        f.write("Performance du modèle Régression Linéaire\n")
        f.write(f"RMSE: {lin_rmse:.4f} secondes\n")
        f.write(f"MAE: {lin_mae:.4f} secondes\n")
        f.write("--------------------------------------------------\n")

    print("\nTerminé avec succès !")


if __name__ == "__main__":
    main()
