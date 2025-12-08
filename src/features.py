import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(df):
    """
    Prépare les features (X) et la cible (y) pour le modèle.
    """
    # Encodage du type de pneu (Ordinal est acceptable ici car Soft < Medium < Hard)
    # Mapping explicite pour être sûr de l'ordre
    compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
    df['Compound_Encoded'] = df['Compound'].map(compound_map)

    # Encodage du circuit (si plusieurs circuits)
    # On utilise LabelEncoder pour transformer les noms de circuits en ID
    le_circuit = LabelEncoder()
    df['Circuit_ID'] = le_circuit.fit_transform(df['Circuit'])

    # Sélection des Features (X)
    # On utilise TyreLife (usure), Circuit, Compound, et LapNumber (proxy carburant)
    # On ajoute TrackTemp car la chaleur dégrade les pneus
    features = ['TyreLife', 'Compound_Encoded',
                'LapNumber', 'Circuit_ID', 'TrackTemp']

    X = df[features]
    y = df['LapTime_Sec']  # Ce qu'on veut prédire

    return X, y, le_circuit


def split_data(X, y, test_size=0.2):
    """
    Sépare les données en ensemble d'entraînement et de test.
    random_state=42 assure que les résultats sont reproductibles (important pour le rapport).
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)
