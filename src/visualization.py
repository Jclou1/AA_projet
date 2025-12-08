import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_predictions(y_test, predictions, title="Réalité vs Prédictions"):
    """
    Affiche un graphique de dispersion pour voir si les prédictions collent à la réalité.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)

    # Ligne parfaite (y=x)
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.xlabel("Temps Réel (sec)")
    plt.ylabel("Temps Prédit (sec)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    rand = np.random.randint(0, 10000)
    plt.savefig(f"outputs/resultats_model_{rand}.png")
    print(f"Graphique sauvegardé sous 'outputs/resultats_model_{rand}.png'")
    # plt.show()


def plot_degradation_curve(model, circuit_id, track_temp=30, title="Courbe de Dégradation des Pneus"):
    """
    Simule une courbe de dégradation pour l'affiche (Soft vs Medium vs Hard).
    Supposons un tour 1 à 30 avec une température fixe.
    """

    laps = np.arange(1, 40)  # 40 tours

    # On crée des données fictives pour la simulation
    # 0=Soft, 1=Medium, 2=Hard
    compounds = {'Soft': 0, 'Medium': 1, 'Hard': 2}

    plt.figure(figsize=(12, 6))

    for name, code in compounds.items():
        # Création des features pour la prédiction
        # ['TyreLife', 'Compound_Encoded', 'LapNumber', 'Circuit_ID', 'TrackTemp']
        # On assume que TyreLife = LapNumber pour un relais depuis le départ
        X_sim = pd.DataFrame({
            'TyreLife': laps,
            'Compound_Encoded': [code] * len(laps),
            'LapNumber': laps,
            'Circuit_ID': [circuit_id] * len(laps),
            'TrackTemp': [track_temp] * len(laps)
        })

        preds = model.predict(X_sim)
        plt.plot(laps, preds, label=f'{name} Compound', linewidth=2)

    plt.xlabel("Âge du pneu (Tours)")
    plt.ylabel("Temps au tour estimé (sec)")
    plt.title(f"{title} (Temp Piste: {track_temp}°C)")
    plt.legend()
    plt.grid(True)
    rand = np.random.randint(0, 10000)
    plt.savefig(f"outputs/simulation_degradation_{rand}.png")
    print(
        f"Graphique sauvegardé sous 'outputs/simulation_degradation_{rand}.png'")


def plot_degradation_curve(model, circuit_id, track_temp=35, title="Courbe de Dégradation des Pneus"):
    """
    Simule une courbe de dégradation en isolant l'usure des pneus.
    On fixe le 'LapNumber' (poids essence) pour ne voir que l'effet gomme.
    """

    # On simule une usure de 1 à 30 tours
    laps = np.arange(1, 31)

    # On fixe le LapNumber à 25 (milieu de course) pour tout le monde.
    # Ainsi, le modèle ne voit pas la voiture s'alléger par la diminution du carburant.
    fixed_fuel_lap = 25

    compounds = {'Soft': 0, 'Medium': 1, 'Hard': 2}
    colors = {'Soft': 'red', 'Medium': 'yellow',
              'Hard': 'black'}  # Couleurs F1 standards

    plt.figure(figsize=(10, 6))

    for name, code in compounds.items():
        # Création des données de simulation
        X_sim = pd.DataFrame({
            'TyreLife': laps,
            'Compound_Encoded': [code] * len(laps),

            'LapNumber': [fixed_fuel_lap] * len(laps),

            'Circuit_ID': [circuit_id] * len(laps),
            'TrackTemp': [track_temp] * len(laps)
        })

        preds = model.predict(X_sim)

        plt.plot(laps, preds, label=f'{name}',
                 linewidth=2, color=colors.get(name, 'blue'))

    plt.xlabel("Âge du pneu (Tours)")
    plt.ylabel("Temps au tour estimé (sec) - À iso-carburant")
    plt.title(
        f"{title} (Poids fixe: Tour {fixed_fuel_lap}, Temp Piste: {track_temp}°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sauvegarde
    rand = np.random.randint(0, 10000)
    plt.savefig(f"outputs/courbe_degradation_{rand}.png")
    print(
        f"Nouvelle courbe générée : outputs/courbe_degradation_{rand}.png")
