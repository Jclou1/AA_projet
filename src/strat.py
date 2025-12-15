import numpy as np
import pandas as pd
from scipy import stats

COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
REVERSE_COMPOUND_MAP = {v: k for k, v in COMPOUND_MAP.items()}


def clean_predictions(predictions, window_size=5):
    """
    Applique un filtre de mode (vote majoritaire) pour lisser les prédictions
    et éviter les changements de pneus rapides et irréalistes.
    """
    # On utilise une fenêtre glissante pour trouver la valeur la plus fréquente locale
    # window_size=5 signifie qu'on regarde 2 tours avant et 2 tours après
    series = pd.Series(predictions)

    # Rolling mode n'existe pas directement de façon optimisée dans pandas,
    # mais pour des petites séries, ceci fonctionne bien :
    smooth_preds = series.rolling(window=window_size, center=True, min_periods=1).apply(
        lambda x: stats.mode(x, keepdims=False)[0]
    ).ffill().bfill()

    return smooth_preds.values.astype(int)


def parse_strategy(X, compound, clean=True):
    """
    Parse predicted tyre strategies per race.
    Option 'clean' active le lissage.
    """
    results = {}

    # 1. Récupération des prédictions brutes
    raw_compounds = compound.values.astype(int)

    # 2. NETTOYAGE : On lisse les prédictions pour éviter les arrêts inutiles
    if clean:
        compounds = clean_predictions(raw_compounds, window_size=5)
    else:
        compounds = raw_compounds

    # (Le reste de votre logique reste identique, mais on utilise 'compounds' nettoyé)

    # Pour le test actuel (Abu Dhabi 2025)
    race = "Abu Dhabi"

    strategy = []
    if len(compounds) > 0:
        current_compound = REVERSE_COMPOUND_MAP.get(compounds[0], "UNKNOWN")
        stint_length = 1

        for i in range(1, len(compounds)):
            pred_compound_name = REVERSE_COMPOUND_MAP.get(
                compounds[i], "UNKNOWN")

            if pred_compound_name == current_compound:
                stint_length += 1
            else:
                strategy.append((current_compound, stint_length))
                current_compound = pred_compound_name
                stint_length = 1

        # Add the last stint
        strategy.append((current_compound, stint_length))

    results[race] = strategy

    return results
