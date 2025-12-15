import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

COMPOUND_MAP = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4
}

custom_palette = ["#121F45", "#223971", "#6674A3", "#CC1E4A", "#FFC906"]

COMPOUND_COLORS = {
    "SOFT": "red",
    "MEDIUM": "gold",
    "HARD": "black",
}


def plot_actual_strat_vs_predicted_strat(actual_strat, results, driver, circuit_name, model_name="RandomForest"):
    """
    Trace les stratégies pneus par tour pour chaque circuit.
    X = numéro de tour
    Y = compound (0=SOFT, 1=MEDIUM, ...)
    Un graphique par circuit.

    Params:
        actual_strat (dict): {circuit: [(compound_name, length), ...], ...}
        results (pd.DataFrame): DataFrame avec colonnes 'Model' et 'Strat'
        model_name (str): modèle à afficher
    """
    predicted_strat = results.loc[results['Model']
                                  == model_name, 'Strat'].values[0]

    for circuit in actual_strat.keys():
        plt.figure(figsize=(12, 6))

        actual_list = actual_strat[circuit]
        actual_laps = []
        actual_compounds = []
        lap_counter = 1
        for compound, length in actual_list:
            y_val = COMPOUND_MAP[compound]
            laps = list(range(lap_counter, lap_counter + length))
            actual_laps.extend(laps)
            actual_compounds.extend([y_val]*length)
            lap_counter += length

        predicted_list = predicted_strat.get(circuit, [])
        pred_laps = []
        pred_compounds = []
        lap_counter = 1
        for compound, length in predicted_list:
            y_val = COMPOUND_MAP[compound]
            laps = list(range(lap_counter, lap_counter + length))
            pred_laps.extend(laps)
            pred_compounds.extend([y_val]*length)
            lap_counter += length

        plt.scatter(actual_laps, actual_compounds,
                    color='blue', label='Réel', alpha=0.7)
        plt.scatter(pred_laps, pred_compounds, color='orange',
                    marker='x', label='Prédit', alpha=0.8)

        plt.xlabel("Tour")
        plt.ylabel("Pneu (compound)")
        plt.title(
            f"Abu_Dhabi - Stratégie pneus ({model_name}, Pilote: {driver})")
        plt.yticks(range(3), ["SOFT", "MEDIUM", "HARD"])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = f"outputs/drivers/strat_{circuit_name}_2025_{model_name}_{driver}.png"
        plt.savefig(filename)
        print(f"Graphique sauvegardé sous '{filename}'")


def plot_accuracy_comparison(average_accuracy_per_model):
    """
    Trace un graphique comparant les précisions moyennes par modèle.
    """
    models = list(average_accuracy_per_model.keys())
    avg_accuracies = [np.mean(average_accuracy_per_model[m]) for m in models]

    plt.figure(figsize=(10, 6))
    # Pass the custom palette with custom hex colors
    sns.barplot(x=models, y=avg_accuracies,
                hue=avg_accuracies, palette=custom_palette)

    plt.xlabel("Modèle")
    plt.ylabel("Précision moyenne")
    plt.title("Comparaison des précisions moyennes par modèle")
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)

    rand = np.random.randint(0, 10000)
    filename = f"outputs/accuracy_comparison_{rand}.png"
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Graphique de comparaison des précisions sauvegardé : {filename}")


def plot_aggregated_feature_importance(importances, feature_names, title="Importance moyenne des features (Random Forest)"):
    """
    Affiche l'importance moyenne des variables sur l'ensemble des pilotes.
    """
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature',
                data=df_imp, hue='Feature', palette=custom_palette)

    plt.title(title)
    plt.xlabel("Poids Moyen (Importance)")
    plt.ylabel("Variables")
    plt.grid(axis='x', alpha=0.3)

    for index, value in enumerate(df_imp['Importance']):
        plt.text(value, index, f'{value:.3f}', va='center', fontsize=9)

    plt.tight_layout()

    rand = np.random.randint(0, 10000)
    filename = f"outputs/avg_feature_importance_{rand}.png"
    plt.savefig(filename)
    print(f"Graphique moyen sauvegardé : {filename}")


def plot_accuracy_trend_by_data_size(trend_data):
    """
    Affiche l'évolution de la précision en fonction du nombre d'années d'entraînement.

    trend_data est une liste de dictionnaires :
    [
        {'label': '2019', 'scores': {'RandomForest': 0.75, 'Logistic': 0.70}},
        {'label': '2019-2020', 'scores': {'RandomForest': 0.78, 'Logistic': 0.72}},
        ...
    ]
    """
    plot_data = []
    for entry in trend_data:
        label = entry['label']
        for model_name, score in entry['scores'].items():
            plot_data.append({
                'Données': label,
                'Précision': score,
                'Modèle': model_name
            })

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 6))

    sns.lineplot(data=df_plot, x='Données',
                 y='Précision', hue='Modèle', marker='o', linewidth=2.5)

    plt.title(
        "Impact du volume de données historiques sur la précision (Test: Abu Dhabi 2025)")
    plt.ylabel("Précision moyenne (Accuracy)")
    plt.xlabel("Années incluses dans l'entraînement")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    rand = np.random.randint(0, 10000)
    filename = f"outputs/learning_curve_data_size_{rand}.png"
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Graphique de tendance sauvegardé : {filename}")
