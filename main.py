from src.data_loader import load_multiple_races, load_race_data, setup_cache
from src.features import prepare_data
from src.models import train_all_models, evaluate_all_models
from src.visualization import (
    plot_actual_strat_vs_predicted_strat,
    plot_accuracy_comparison,
    plot_aggregated_feature_importance,
    plot_accuracy_trend_by_data_size
)
from src.strat import parse_strategy
import numpy as np
import pandas as pd

from fastf1 import logger
logger.set_log_level("ERROR")

setup_cache()


def main():
    print("🏎️  F1 Tire Degradation Predictor")
    print("Analyse Complète : Stratégies & Courbe d'Apprentissage\n")

    # Liste des années historiques pour l'entraînement
    # On ajoute les années progressivement pour voir l'impact du volume de données
    full_races_config = [
        (2019, 'Abu Dhabi'),
        (2020, 'Abu Dhabi'),
        (2021, 'Abu Dhabi'),
        (2022, 'Abu Dhabi'),
        (2023, 'Abu Dhabi'),
        (2024, 'Abu Dhabi'),
    ]

    target_drivers = ['VER', 'LEC', 'HAM', 'GAS', 'RUS', 'NOR']
    test_year = 2025
    test_gp = 'Abu Dhabi'

    # On charge 2025 une seule fois pour gagner du temps
    print(f"📥 Chargement du jeu de test cible ({test_year} {test_gp})...")
    df_test_full = load_race_data(test_year, test_gp)

    if df_test_full.empty:
        print("Erreur : Impossible de charger le jeu de test.")
        return

    # Pour la courbe d'apprentissage (Learning Curve)
    trend_results = []

    # Pour l'analyse finale (Bar charts & Feature Importance)
    final_avg_accuracy = {}  # {Model: [acc_driver1, acc_driver2...]}
    final_feature_importances = []
    feature_names = None

    # On commence avec 1 année, puis 2, puis 3... jusqu'à tout le dataset.

    for i in range(1, len(full_races_config) + 1):
        # Définition du sous-ensemble d'entraînement
        subset_config = full_races_config[:i]

        if i == 1:
            years_label = str(subset_config[0][0])
        else:
            years_label = f"{subset_config[0][0]}-{subset_config[-1][0]}"

        is_final_run = (i == len(full_races_config))

        print(
            f"\n🔄 [Itération {i}/{len(full_races_config)}] Entraînement sur : {years_label}")

        # Chargement des données d'entraînement pour ce sous-ensemble
        df_train_full = load_multiple_races(subset_config)
        if df_train_full.empty:
            continue

        # Stockage temporaire pour moyenner les scores des pilotes pour cette étape
        current_step_accuracies = {}

        # Boucle sur chaque pilote cible
        for driver in target_drivers:
            # Filtrage des données pour ce pilote
            df_train = df_train_full[df_train_full['Driver'] == driver]
            df_test = df_test_full[df_test_full['Driver'] == driver]

            if df_train.empty or df_test.empty:
                continue

            # Préparation (X, y)
            X_train, y_train = prepare_data(df_train)
            X_test, y_test = prepare_data(df_test)

            # Entraînement des modèles
            models = train_all_models(X_train, y_train)

            # Évaluation
            results = evaluate_all_models(models, X_test, y_test)

            # Reconstruction de la stratégie (utile pour le final run)
            actual_strat = parse_strategy(X_test, y_test)

            # Collecte des scores pour la courbe d'apprentissage
            for model_name in results['Model'].unique():
                acc = results.loc[results['Model'] ==
                                  model_name, 'Accuracy'].values[0]

                if model_name not in current_step_accuracies:
                    current_step_accuracies[model_name] = []
                current_step_accuracies[model_name].append(acc)

            # Actions Spécifiques pour la dernière itération
            if is_final_run:
                print(f"   👤 {driver} traité (Full Data).")

                # Sauvegarde pour Accuracy Bar Chart
                for model_name in results['Model'].unique():
                    acc = results.loc[results['Model'] ==
                                      model_name, 'Accuracy'].values[0]
                    if model_name not in final_avg_accuracy:
                        final_avg_accuracy[model_name] = []
                    final_avg_accuracy[model_name].append(acc)

                # Sauvegarde pour Feature Importance (Random Forest uniquement)
                if 'RandomForest' in models:
                    feature_names = X_train.columns.tolist()
                    final_feature_importances.append(
                        models['RandomForest'].feature_importances_)

                    # Génération du graphique de Stratégie Individuelle
                    # On affiche explicitement la dernière course d'entraînement comme référence dans le titre
                    last_train_gp = subset_config[-1][1]
                    plot_actual_strat_vs_predicted_strat(
                        actual_strat, results, driver, circuit_name=last_train_gp, model_name="RandomForest"
                    )

        # Calcul de la moyenne pour l'étape courante
        if current_step_accuracies:
            # Moyenne de l'accuracy de tous les pilotes pour cette taille de dataset
            step_avg_scores = {m: np.mean(
                scores) for m, scores in current_step_accuracies.items()}

            trend_results.append({
                'label': years_label,
                'scores': step_avg_scores
            })

            # Petit log pour suivre la progression
            best_model_step = max(step_avg_scores, key=step_avg_scores.get)
            print(
                f"   📈 Moyenne globale ({years_label}) - {best_model_step}: {step_avg_scores[best_model_step]:.2%}")

    print("\n\n=== 📊 Génération des Rapports Visuels ===")

    # Courbe d'apprentissage
    if trend_results:
        print("1. Génération de la Courbe d'Apprentissage...")
        plot_accuracy_trend_by_data_size(trend_results)

    # Importance Moyenne des Features (tous les pilotes)
    if final_feature_importances and feature_names:
        print("2. Génération de l'Importance Moyenne des Features...")
        avg_importances = np.mean(final_feature_importances, axis=0)
        plot_aggregated_feature_importance(avg_importances, feature_names)

    # Comparaison des modèles
    if final_avg_accuracy:
        print("3. Génération de la Comparaison des Modèles...")
        # Affichage console des moyennes finales
        print("\n--- Précision Moyenne Finale (Toutes années) ---")
        for model, accuracies in final_avg_accuracy.items():
            avg_acc = np.mean(accuracies)
            print(f"{model:<20} : {avg_acc:.2%}")

        plot_accuracy_comparison(final_avg_accuracy)

    print("\n✅ Analyse terminée avec succès !")


if __name__ == "__main__":
    main()
