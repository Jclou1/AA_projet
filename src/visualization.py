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
    
def plot_tyre_degradation_by_circuit(df, circuit_name, title=None):
    """
    Visualise la dégradation des pneus sur un circuit donné,
    en comparant plusieurs années / courses.

    Paramètres:
      - df : DataFrame global (celui venant de load_multiple_races)
              doit contenir au moins :
                ['Circuit', 'Year', 'TyreLife', 'LapTime_Sec', 'TrackTemp']
      - circuit_name : nom du circuit tel que stocké dans df['Circuit']
                       (ex: 'Bahrain', 'Saudi Arabia', etc.)
    """

    # Filtre sur le circuit demandé
    data = df[df['Circuit'] == circuit_name].copy()

    if data.empty:
        print(f"Aucune donnée trouvée pour le circuit '{circuit_name}'")
        return

    # Si pas de titre fourni, on en met un par défaut
    if title is None:
        title = f"Dégradation des pneus sur {circuit_name} (comparaison par année)"

    years = sorted(data['Year'].unique())

    plt.figure(figsize=(10, 6))

    # On boucle sur chaque année pour les distinguer par marqueur/couleur
    markers = ['o', 's', '^', 'D', 'x', 'P', 'v', '*']
    import itertools
    marker_cycle = itertools.cycle(markers)

    # On va aussi colorier par TrackTemp
    # On garde la même échelle de couleur pour tout le circuit
    temps = data['TrackTemp']
    vmin = temps.min()
    vmax = temps.max()

    for year in years:
        sub = data[data['Year'] == year]

        marker = next(marker_cycle)

        sc = plt.scatter(
            sub['TyreLife'],
            sub['LapTime_Sec'],
            c=sub['TrackTemp'],
            vmin=vmin,
            vmax=vmax,
            alpha=0.6,
            marker=marker,
            label=str(year)
        )

    cbar = plt.colorbar(sc)
    cbar.set_label("Température piste (°C)")

    plt.xlabel("Usure du pneu (TyreLife, en tours)")
    plt.ylabel("Temps au tour (sec)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Année", loc="best")

    plt.tight_layout()

    import numpy as np
    rand = np.random.randint(0, 10000)
    filename = f"outputs/tyre_deg_{circuit_name.lower()}_{rand}.png"
    plt.savefig(filename)
    print(f"Graphe dégradation par circuit sauvegardé dans '{filename}'")


def plot_mean_laptime_vs_tyre_life_by_temp(df, circuit_name=None):
    """
    Tendance de l'usure pour différentes plages de température de piste.
    """

    data = df.copy()
    if circuit_name is not None:
        data = data[data["Circuit"] == circuit_name]

    data["TempBin"] = pd.cut(
        data["TrackTemp"],
        bins=[0, 25, 35, 50],
        labels=["Froid", "Modéré", "Chaud"]
    )

    data["TyreLifeBin"] = data["TyreLife"].astype(int)

    plt.figure(figsize=(10, 6))

    for label in data["TempBin"].dropna().unique():
        sub = data[data["TempBin"] == label]
        grouped = sub.groupby("TyreLifeBin")["LapTime_Sec"].mean()
        plt.plot(grouped.index, grouped.values, marker='o', label=str(label))

    plt.xlabel("Usure du pneu (TyreLife)")
    plt.ylabel("Temps moyen au tour (sec)")
    
    if circuit_name:
        plt.title(f"Dégradation des pneus selon la température – {circuit_name}")
    else:
        plt.title("Dégradation des pneus selon la température")

    plt.legend(title="Température piste")
    plt.grid(True, alpha=0.3)

    import numpy as np
    rand = np.random.randint(0, 10000)
    filename = f"outputs/tendance_usure_temp_{rand}.png"
    plt.tight_layout()
    plt.savefig(filename)

    print(f"Graphe tendance usure par température sauvegardé : {filename}")

def analyze_degradation_by_stint(df, circuit_name=None):
    """
    Analyse la dégradation des pneus à l'intérieur de chaque relais (Stint).
    Calcule la pente (sec/tour) de la dégradation pour chaque relais valide
    et génère un graphique sur quelques relais représentatifs.
    """

    data = df.copy()

    if circuit_name is not None:
        data = data[data["Circuit"] == circuit_name]

    # On garde seulement les relais exploitables
    data = data.dropna(subset=["Stint", "TyreLife", "LapTime_Sec"])

    import numpy as np
    import matplotlib.pyplot as plt

    slopes = []
    valid_stints = []

    for stint_id in sorted(data["Stint"].unique()):
        sub = data[data["Stint"] == stint_id]

        # Il faut suffisamment de points pour estimer une pente fiable
        if len(sub) >= 6:
            x = sub["TyreLife"].values
            y = sub["LapTime_Sec"].values

            # Régression linéaire sur le relais : y = a*x + b
            a, b = np.polyfit(x, y, 1)
            slopes.append(a)
            valid_stints.append(stint_id)

    if len(slopes) == 0:
        print("Aucun relais valide pour l'analyse.")
        return

    slopes = np.array(slopes)

    # --- Statistiques globales ---
    print("\n--- Analyse de la dégradation par relais ---")
    print(f"Nombre de relais analysés : {len(slopes)}")
    print(f"Pente moyenne de dégradation : {slopes.mean():.4f} sec / tour")
    print(f"Pente minimale : {slopes.min():.4f} sec / tour")
    print(f"Pente maximale : {slopes.max():.4f} sec / tour")
    print(f"Écart-type : {slopes.std():.4f} sec / tour")

    # --- Graphique de quelques relais représentatifs ---
    plt.figure(figsize=(10, 6))

    # On affiche 5 relais bien répartis
    sample_stints = valid_stints[::max(1, len(valid_stints)//5)][:5]

    for stint_id in sample_stints:
        sub = data[data["Stint"] == stint_id]

        x = sub["TyreLife"].values
        y = sub["LapTime_Sec"].values

        a, b = np.polyfit(x, y, 1)
        y_fit = a * x + b

        plt.plot(x, y, 'o', alpha=0.4)
        plt.plot(x, y_fit, label=f"Stint {stint_id} (pente={a:.3f})")

    plt.xlabel("Usure du pneu (TyreLife)")
    plt.ylabel("Temps au tour (sec)")
    
    if circuit_name:
        plt.title(f"Dégradation des pneus par relais – {circuit_name}")
    else:
        plt.title("Dégradation des pneus par relais")

    plt.legend()
    plt.grid(True, alpha=0.3)

    rand = np.random.randint(0, 10000)
    filename = f"outputs/degradation_par_relais_{rand}.png"
    plt.tight_layout()
    plt.savefig(filename)

    print(f"Graphe par relais sauvegardé dans : {filename}")

    return slopes


def analyze_degradation_by_stint_and_compound(df, circuit_name=None):
    """
    Calcule la pente de dégradation (sec/tour) par relais,
    séparée par type de pneu (Soft / Medium / Hard),
    et génère des graphiques comparatifs.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    data = df.copy()

    if circuit_name is not None:
        data = data[data["Circuit"] == circuit_name]

    data = data.dropna(subset=["Stint", "TyreLife", "LapTime_Sec", "Compound"])

    results = {
        "SOFT": [],
        "MEDIUM": [],
        "HARD": []
    }

    # --- Calcul des pentes par relais ---
    for stint_id in sorted(data["Stint"].unique()):
        sub = data[data["Stint"] == stint_id]

        if len(sub) >= 6:  # assez de points
            x = sub["TyreLife"].values
            y = sub["LapTime_Sec"].values
            a, b = np.polyfit(x, y, 1)

            compound = sub["Compound"].iloc[0]
            if compound in results:
                results[compound].append(a)

    # --- Statistiques ---
    print("\n--- Analyse de la dégradation par relais et par type de pneu ---")
    for comp, slopes in results.items():
        if len(slopes) > 0:
            slopes = np.array(slopes)
            print(f"\n{comp}")
            print(f"  Nombre de relais : {len(slopes)}")
            print(f"  Pente moyenne    : {slopes.mean():.4f} sec/tour")
            print(f"  Pente min        : {slopes.min():.4f}")
            print(f"  Pente max        : {slopes.max():.4f}")
            print(f"  Écart-type       : {slopes.std():.4f}")
            results[comp] = slopes
        else:
            print(f"\n{comp} : aucun relais valide")

    # --- BOXPLOT COMPARATIF ---
    plt.figure(figsize=(9, 6))

    data_to_plot = []
    labels = []

    for comp in ["SOFT", "MEDIUM", "HARD"]:
        if len(results[comp]) > 0:
            data_to_plot.append(results[comp])
            labels.append(comp)

    plt.boxplot(data_to_plot, labels=labels, showfliers=True)
    plt.ylabel("Pente de dégradation (sec / tour)")
    if circuit_name:
        plt.title(f"Comparaison de la dégradation par type de pneu – {circuit_name}")
    else:
        plt.title("Comparaison de la dégradation par type de pneu")

    plt.grid(True, alpha=0.3)

    import numpy as np
    rand = np.random.randint(0, 10000)
    filename1 = f"outputs/pentes_par_compose_{rand}.png"
    plt.tight_layout()
    plt.savefig(filename1)

    print(f"Boxplot sauvegardé : {filename1}")

    # --- HISTOGRAMMES ---
    plt.figure(figsize=(10, 6))

    for comp in ["SOFT", "MEDIUM", "HARD"]:
        if len(results[comp]) > 0:
            plt.hist(results[comp], bins=12, alpha=0.6, label=comp)

    plt.xlabel("Pente de dégradation (sec / tour)")
    plt.ylabel("Nombre de relais")
    if circuit_name:
        plt.title(f"Distribution des pentes par type de pneu – {circuit_name}")
    else:
        plt.title("Distribution des pentes par type de pneu")

    plt.legend()
    plt.grid(True, alpha=0.3)

    filename2 = f"outputs/histogramme_pentes_{rand}.png"
    plt.tight_layout()
    plt.savefig(filename2)

    print(f"Histogramme sauvegardé : {filename2}")

    return results

def plot_model_performance(results, save_path="outputs/model_performance_models.png"):
    """
    Affiche un graphique comparant la RMSE de chaque modèle.

    results : dict {nom_modèle: {"rmse": ..., "mae": ..., "preds": ...}}
               (ce que retourne evaluate_all_models)
    """
    model_names = results['Model'].tolist()
    rmses = [results.loc[results['Model'] == m, "RMSE"].values[0] for m in model_names]

    # On trouve le meilleur modèle pour le mettre en avant
    best_idx = min(range(len(model_names)), key=lambda i: rmses[i])

    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(len(model_names)), rmses)

    # Met le meilleur modèle en vert
    bars[best_idx].set_color("green")

    plt.xticks(range(len(model_names)), model_names, rotation=20, ha="right")
    plt.ylabel("RMSE (s)")
    plt.title("Comparaison des modèles – RMSE sur l'ensemble de test")

    # Affiche la valeur numérique au-dessus de chaque barre
    for i, val in enumerate(rmses):
        plt.text(i, val + 0.03, f"{val:.2f}", ha="center", fontsize=9)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Graphique de performance des modèles sauvegardé : {save_path}")

COMPOUND_COLORS = {
    "SOFT": "red",
    "MEDIUM": "gold",
    "HARD": "black",
}

def plot_degradation_panels(
    df,
    model,
    feature_template,
    feature_names,
    le_circuit,
    circuit_name,
    track_temp=35,
    lap_number=14,
    max_tyre_life=None,
    save_path="outputs/degradation_panels.png",
):
    """
    3 sous-graphes (SOFT/MEDIUM/HARD) :
      - nuage de points gris = données brutes
      - ligne pleine = moyenne des données par TyreLife
      - ligne pointillée = modèle ML

    C'est beaucoup plus lisible que tout empilé dans un seul graphe.
    """

    lap_col = "LapTime_Sec"

    # 1) Filtrer sur le circuit
    if "Circuit" not in df.columns:
        print("❌ Colonne 'Circuit' absente.")
        print("Colonnes :", list(df.columns))
        return

    df_circ = df[df["Circuit"] == circuit_name].copy()
    if df_circ.empty:
        print(f"❌ Aucune donnée pour le circuit '{circuit_name}'")
        return

    # Convertir en secondes si besoin
    lap_series = df_circ[lap_col]
    if np.issubdtype(lap_series.dtype, np.dtype("timedelta64[ns]")):
        df_circ["_LapTime_sec"] = lap_series.dt.total_seconds()
    else:
        df_circ["_LapTime_sec"] = lap_series.astype(float)

    # Encodage du circuit
    circuit_id = le_circuit.transform([circuit_name])[0]

    compounds = ["SOFT", "MEDIUM", "HARD"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, comp in zip(axes, compounds):
        if "Compound" not in df_circ.columns or "TyreLife" not in df_circ.columns:
            print("❌ Colonnes 'Compound' ou 'TyreLife' manquantes.")
            return

        df_comp = df_circ[df_circ["Compound"] == comp].copy()
        if df_comp.empty:
            ax.set_title(f"{comp} (aucune donnée)")
            continue

        if max_tyre_life is not None:
            df_comp = df_comp[df_comp["TyreLife"] <= max_tyre_life]

        # Nuage de points brut (gris)
        ax.scatter(
            df_comp["TyreLife"],
            df_comp["_LapTime_sec"],
            alpha=0.15,
            s=8,
            color="grey",
            label="Données brutes" if comp == "SOFT" else None,
        )

        # Moyenne + écart-type par TyreLife
        stats = (
            df_comp.groupby("TyreLife")["_LapTime_sec"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        ages = stats["TyreLife"].values
        mean_data = stats["mean"].values
        std_data = stats["std"].values

        ax.plot(
            ages,
            mean_data,
            linewidth=2,
            color=COMPOUND_COLORS.get(comp, None),
            label="Moyenne données" if comp == "SOFT" else None,
        )
        ax.fill_between(
            ages,
            mean_data - std_data,
            mean_data + std_data,
            alpha=0.15,
            color=COMPOUND_COLORS.get(comp, None),
        )

        # Courbe du modèle pour les mêmes ages
        preds = []
        mapping = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
        for age in ages:
            row = feature_template.copy()
            if "TyreLife" in row.index:
                row["TyreLife"] = age
            if "Compound_Encoded" in row.index:
                row["Compound_Encoded"] = mapping[comp]
            if "LapNumber" in row.index:
                row["LapNumber"] = lap_number
            if "Circuit_ID" in row.index:
                row["Circuit_ID"] = circuit_id
            if "TrackTemp" in row.index:
                row["TrackTemp"] = track_temp

            X_row = pd.DataFrame([row])[list(feature_names)]
            preds.append(model.predict(X_row)[0])

        ax.plot(
            ages,
            preds,
            linestyle="--",
            linewidth=2,
            color=COMPOUND_COLORS.get(comp, None),
            label="Modèle" if comp == "SOFT" else None,
        )

        ax.set_title(comp)
        ax.set_xlabel("TyreLife (tours)")

    axes[0].set_ylabel("Temps au tour (s)")

    # Légende commune
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)

    fig.suptitle(
        f"Dégradation des pneus – {circuit_name}\n"
        f"Temp piste : {track_temp}°C | Lap ref : {lap_number}",
        y=1.03,
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Graphique panneaux sauvegardé : {save_path}")


def plot_actual_strat_vs_predicted_strat(actual_strat, results, model_name="RandomForest"):
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
    # Récupérer la stratégie prédite pour le modèle
    predicted_strat = results.loc[results['Model'] == model_name, 'Strat'].values[0]
    
    for circuit in actual_strat.keys():
        plt.figure(figsize=(12,6))
        
        # --- Réel ---
        actual_list = actual_strat[circuit]
        actual_laps = []
        actual_compounds = []
        lap_counter = 1
        for compound, length in actual_list:
            laps = list(range(lap_counter, lap_counter + length))
            actual_laps.extend(laps)
            actual_compounds.extend([compound]*length)
            lap_counter += length
        
        # --- Prédit ---
        predicted_list = predicted_strat.get(circuit, [])
        pred_laps = []
        pred_compounds = []
        lap_counter = 1
        for compound, length in predicted_list:
            laps = list(range(lap_counter, lap_counter + length))
            pred_laps.extend(laps)
            pred_compounds.extend([compound]*length)
            lap_counter += length
        
        # Scatter plot
        plt.scatter(actual_laps, actual_compounds, color='blue', label='Réel', alpha=0.7)
        plt.scatter(pred_laps, pred_compounds, color='orange', marker='x', label='Prédit', alpha=0.8)
        
        plt.xlabel("Tour")
        plt.ylabel("Pneu (compound)")
        plt.title(f"{circuit} - Stratégie pneus: réel vs prédit ({model_name})")
        plt.yticks(range(5), ["SOFT","MEDIUM","HARD","INTERMEDIATE","WET"])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        filename = f"outputs/strat_{circuit}_{model_name}.png"
        plt.savefig(filename)
        print(f"Graphique sauvegardé sous '{filename}'")
        plt.show()