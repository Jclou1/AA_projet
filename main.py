from src.data_loader import load_multiple_races, build_races_config_for_circuit
from src.features import prepare_data, split_data
from src.models import train_models, evaluate_models, feature_importance
from src.visualization import plot_predictions, plot_degradation_curve, plot_tyre_degradation_by_circuit, plot_mean_laptime_vs_tyre_life_by_temp, analyze_degradation_by_stint, analyze_degradation_by_stint_and_compound
from src.strat import generate_strategies, simulate_strategy
import matplotlib.pyplot as plt
def main():
    print("F1 Tire Degradation Predictor")

    # Listes des courses à analyser
    # Bahrain est très abrasif (bon pour voir l'usure)
#    races_config = [
#        (2024, 'Bahrain'),
#        (2024, 'Saudi Arabia'),
        # (2024, 'Australia'),
        # (2024, 'Italy'),
        # (2024, 'Singapore'),
        # (2024, 'Japan'),
        # (2024, 'USA'),
        # (2024, 'Mexico'),
        # (2024, 'Brazil'),
        # (2024, 'Abu Dhabi'),
        # (2024, 'Monaco'),
        # (2024, 'Canada'),
#    ]

    races_config = build_races_config_for_circuit(
    circuit_keyword="Bahrain",
    start_year=2018,
    end_year=2024
)

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

        # Courbe physique : dégradation des pneus en fonction de la température
    #plot_tyre_degradation_by_circuit(
    #    df,
    #    circuit_name='Bahrain',
    #    title="Dégradation des pneus à Bahrain (comparaison multi-années)"
    #)

    #plot_mean_laptime_vs_tyre_life_by_temp(df, circuit_name="Bahrain")

    #slopes = analyze_degradation_by_stint(df, circuit_name="Bahrain")

    #results = analyze_degradation_by_stint_and_compound(df, circuit_name="Bahrain")

    # Entraînement
    print("\nEntraînement du modèle")
    rf_model, linear_model = train_models(X_train, y_train)

        # Importance des features (Random Forest)
    feature_names = X.columns.tolist()
    feature_importance(rf_model, feature_names)

    # Évaluation
    print("\n Évaluation")
    rf_preds, rf_rmse, rf_mae, lin_preds, lin_rmse, lin_mae = evaluate_models(
        rf_model, linear_model, X_test, y_test)


    print("\n--- Simulation des stratégies de pit stop ---")

    strategies = generate_strategies(total_laps=57)

    circuit_name = "Bahrain"
    circuit_id = le_circuit.transform([circuit_name])[0]

    results = {}

    total_laps=57
    track_temp=35
    pit_time=22

    for name, strat in strategies.items():
        time_total = simulate_strategy(
            rf_model,
            strat,
            total_laps,
            circuit_id,
            track_temp,
            pit_time
        )
        results[name] = time_total
        print(f"{name} : {time_total/60:.2f} minutes")

    best_strategy = min(results, key=results.get)
    print("\n✅ Meilleure stratégie prédite :", best_strategy)

    names = list(results.keys())
    times = [results[n] / 60 for n in names]

    sorted_results = sorted(results.items(), key=lambda x: x[1])

    # On garde seulement le Top 10
    top10 = sorted_results[:10]
    best_time = top10[0][1]  # temps total de la meilleure stratégie (en secondes)

    names = [name for name, t in top10]
    perc_slower = [((t - best_time) / best_time) * 100 for name, t in top10]
    
    plt.figure(figsize=(11, 6))

    bars = plt.bar(range(len(names)), perc_slower)

    plt.xticks(range(len(names)), names, rotation=25, ha="right")
    plt.ylabel("Écart par rapport à la meilleure stratégie (%)")

    title_main = "Top 10 des stratégies de pneus – Écart relatif (Simulation ML)"
    title_details = (
        f"Circuit : {circuit_name} | "
        f"Tours : {total_laps} | "
        f"Température piste : {track_temp}°C | "
        f"Temps de pit : {pit_time}s"
    )

    plt.title(title_main, fontsize=13)
    plt.suptitle(title_details, fontsize=10, y=0.94)

    # Meilleure stratégie en vert
    bars[0].set_color("green")

    # Valeurs au-dessus de chaque barre
    for i, val in enumerate(perc_slower):
        plt.text(i, val + 0.01, f"{val:.2f}%", ha="center", fontsize=9)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/strategies_top10_percent_annotated_full.png")
    print("Graphique Top 10 avec métadonnées complètes sauvegardé : outputs/strategies_top10_percent_annotated_full.png")

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
