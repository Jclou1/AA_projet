import fastf1
import pandas as pd
import numpy as np
import os
from fastf1.core import Laps

# Configuration de l'affichage pandas pour le débogage
pd.set_option('display.max_columns', None)


def setup_cache(cache_dir='.fastf1_cache'):
    """
    Active la cache de FastF1 pour éviter de retélécharger les données à chaque exécution.
    Crée le dossier s'il n'existe pas.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Activation du cache FastF1
    fastf1.Cache.enable_cache(cache_dir)
    print(f"Cache FastF1 activé dans : {cache_dir}")


def load_race_data(year, gp_identifier, session_type='R'):
    """
    Charge et nettoie les données d'une session de course spécifique.

    Args:
        year (int): L'année de la saison (ex: 2024).
        gp_identifier (str/int): Nom du GP (ex: 'Bahrain') ou numéro de la manche.
        session_type (str): Type de session ('R' pour course, 'S' pour Sprint).

    Returns:
        pd.DataFrame: DataFrame contenant les tours nettoyés et enrichis (météo).
    """
    print(
        f"Chargement de la session : {year} - {gp_identifier} ({session_type})...")

    try:
        # Chargement de la session via l'API FastF1
        session = fastf1.get_session(year, gp_identifier, session_type)
        session.load()  # Télécharge la télémétrie, météo, etc.

    except Exception as e:
        print(f"Erreur lors du chargement de la session : {e}")
        return pd.DataFrame()

    # Extraction des tours (Laps)
    laps = session.laps
    
    # Filtrage Initial (Crucial pour le ML)
    # On ne veut que les tours représentatifs de la performance réelle des pneus.

    # Garder seulement les drapeaux verts (TrackStatus = '1')
    # Cela élimine les VSC, SC, et drapeaux jaunes qui faussent les temps.
    laps = laps.pick_track_status('1')

    # Garder les tours "rapides" et précis
    # 'pick_quicklaps' enlève les tours d'entrée/sortie des stands.
    # 'is_accurate' valide que le tour est complet et sans anomalies majeures.
    laps = laps.pick_quicklaps()
    laps = laps[laps['IsAccurate'] == True]

    # Filtrage des composés de pneus
    # On garde uniquement les pneus slicks (Soft, Medium, Hard).
    target_compounds = ['SOFT', 'MEDIUM', 'HARD']
    laps = laps[laps['Compound'].isin(target_compounds)]

    # Enrichissement avec la météo
    # La température de la piste est un facteur clé de dégradation.
    # FastF1 permet de lier la météo à chaque tour.
    weather_data = laps.get_weather_data()
    laps = laps.reset_index(drop=True)

    # On ajoute les colonnes météo importantes au DataFrame des tours
    if not weather_data.empty:
        # On s'assure que les index correspondent
        weather_cols = ['AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']
        # Note: get_weather_data retourne un DF aligné avec les laps fournis
        weather_data = weather_data.reset_index(drop=True)

        for col in weather_cols:
            if col in weather_data.columns:
                laps[col] = weather_data[col]

    # Sélection et nettoyage des colonnes finales
    # Conversion du temps au tour en secondes (float) pour le modèle ML
    laps['LapTime_Sec'] = laps['LapTime'].dt.total_seconds()

    cols_to_keep = [
        'Driver',
        'LapNumber',
        'LapTime_Sec',
        'TyreLife',
        'Compound',
        'Team',
        'TrackTemp',
        'AirTemp',
        'Rainfall',
        'Position',
        'SpeedST',
        'Stint'
    ]

    # Vérification que toutes les colonnes existent avant de filtrer
    existing_cols = [c for c in cols_to_keep if c in laps.columns]
    df_clean = laps[existing_cols].copy()

    # Suppression des lignes avec des valeurs manquantes (NaN)
    # Important : Parfois TrackTemp est manquant sur de vieux GP
    df_clean = df_clean.dropna()

    print(
        f"Données chargées et nettoyées : {len(df_clean)} tours valides récupérés.")

    return df_clean


def load_multiple_races(races_config):
    """
    Charge plusieurs courses et les concatène dans un seul DataFrame.

    Args:
        races_config (list of tuples): Liste [(year, gp), (year, gp), ...]

    Returns:
        pd.DataFrame: DataFrame global concaténé.
    """
    setup_cache()
    all_data = []

    for year, gp in races_config:
        df = load_race_data(year, gp)
        # On ajoute une colonne pour identifier le circuit (utile pour le modèle)
        df['Circuit'] = str(gp)
        df['Year'] = year
        all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def build_races_config_for_circuit(circuit_keyword, start_year=2010, end_year=2025):
    """
    Construit automatiquement la liste races_config pour un circuit donné
    à partir du calendrier FastF1.

    Exemple:
        races_config = build_races_config_for_circuit("Bahrain", 2015, 2024)
    """
    fastf1.Cache.enable_cache(".fastf1_cache")

    races_config = []

    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"Calendrier indisponible pour {year} : {e}")
            continue

        for _, row in schedule.iterrows():

            event_name = str(row.get("EventName", ""))
            event_format = str(row.get("EventFormat", ""))

            # On cherche le circuit par mot-clé
            if circuit_keyword.lower() in event_name.lower():
                races_config.append((year, row["EventName"].split(" Grand")[0]))

                print(f"Ajout : {year} - {row['EventName']}")
                break  # 1 seul GP par année

    return races_config

# ---------------------------------------------------------
# Bloc de test (s'exécute seulement si on lance le fichier directement)
if __name__ == "__main__":
    # Test simple sur le GP de Bahreïn 2024
    setup_cache()
    df = load_race_data(2024, 'Bahrain')

    print("\n--- Aperçu des données ---")
    print(df.head())
    print("\n--- Infos ---")
    print(df.info())

