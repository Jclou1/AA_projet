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

    # Extraction des tours
    laps = session.laps
    laps = laps.pick_track_status('1234567', 'any')

    # Filtrage des composés de pneus
    target_compounds = ['SOFT', 'MEDIUM', 'HARD']
    laps = laps[laps['Compound'].isin(target_compounds)]

    # Enrichissement avec la météo
    weather_data = laps.get_weather_data()
    laps = laps.reset_index(drop=True)

    # On retire les sessions avec pluie
    if not weather_data.empty:
        # On s'assure que les index correspondent
        weather_cols = ['AirTemp', 'TrackTemp', 'Rainfall']
        # Check if there was rain, and if so then return empty
        if weather_data['Rainfall'].max() > 0:
            print("Session avec pluie détectée, les données seront ignorées.")
            return pd.DataFrame()

        weather_data = weather_data.reset_index(drop=True)

        for col in weather_cols:
            if col in weather_data.columns:
                laps[col] = weather_data[col]

    # Sélection et nettoyage des colonnes finales
    # Conversion du temps au tour en secondes pour le modèle ML
    laps['LapTime_Sec'] = laps['LapTime'].dt.total_seconds()

    cols_to_keep = [
        'DriverNumber',
        'Driver',
        'Team',
        'LapNumber',
        'TrackStatus',
        'TyreLife',
        'Stint',
        'Compound',
        'LapTime_Sec',
        'TrackTemp',
        'AirTemp',
        'Rainfall',
        'Position'
    ]

    # Vérification que toutes les colonnes existent avant de filtrer
    existing_cols = [c for c in cols_to_keep if c in laps.columns]
    df_clean = laps[existing_cols].copy()

    # add colum for total laps in the session
    df_clean['TotalLaps'] = session.total_laps

    df_clean = df_clean.fillna(0)

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
        if df.empty:
            print(f"Aucune donnée pour {year} - {gp}, passage au suivant.")
            continue
        # On ajoute une colonne pour identifier le circuit (utile pour le modèle)
        df['Circuit'] = str(gp)
        df['Year'] = year
        all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


if __name__ == "__main__":
    # Test simple sur le GP de Bahreïn 2024
    setup_cache()
    df = load_race_data(2024, 'Bahrain')

    print("\n--- Aperçu des données ---")
    print(df.head())
    print("\n--- Infos ---")
    print(df.info())
