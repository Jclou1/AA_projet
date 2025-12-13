import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit


def prepare_data(df):
    """
    Prépare les features (X) et la cible (y) pour le modèle.
    """
    # Encodage du type de pneu (Ordinal est acceptable ici car Soft < Medium < Hard)
    # Mapping explicite pour être sûr de l'ordre
    compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE':3, 'WET':4}
    df = df.copy()
    df['Compound_Encoded'] = df['Compound'].map(compound_map)

    # Encodage du circuit (si plusieurs circuits)
    # On utilise LabelEncoder pour transformer les noms de circuits en ID
    circuit = LabelEncoder()
    df['Circuit_ID'] = circuit.fit_transform(df['Circuit'])
    circuits_map = {index: label for index, label in zip(df['Circuit_ID'], df['Circuit'])}

    team = LabelEncoder()
    df['Team_Encoded'] = team.fit_transform(df['Team'])

    df['Rainfall'] = df['Rainfall'].apply(lambda x: 0 if x else 1)

    # Sélection des Features (X)
    # On utilise TyreLife (usure), Circuit, Compound, et LapNumber (proxy carburant)
    # On ajoute TrackTemp car la chaleur dégrade les pneus
    features = [
        'Circuit_ID',
        'DriverNumber',
        'Team_Encoded',
        'LapNumber',
        'TrackStatus',
        'LapTime_Sec',
        'TrackTemp',
        'AirTemp',
        'Rainfall',
        'Position'
        ]

    X = df[features]
    y = df['Compound_Encoded']  # Ce qu'on veut prédire

    return X, y, circuits_map


def split_data(X, y, group_col):
    """
    Sépare les données en ensemble d'entraînement et de test.
    random_state=42 assure que les résultats sont reproductibles (important pour le rapport).
    """


    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

    groups = X[group_col]  # e.g. "Circuit" or "RaceID"

    train_idx, test_idx = next(splitter.split(X, y, groups))

    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
