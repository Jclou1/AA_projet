import numpy as np
import pandas as pd
import itertools

COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE":3, "WET":4}
REVERSE_COMPOUND_MAP = {v: k for k, v in COMPOUND_MAP.items()}

def parse_strategy(X, compound):
    """
    Parse predicted tyre strategies per race.
    
    Returns:
        {
            "Bahrain": [("SOFT", 15), ("MEDIUM", 20)],
            "Jeddah":  [("HARD", 18), ("SOFT", 27)]
        }
    """
    results = {}

    # Extract race IDs (or Circuit_ID) from X
    race_ids = X["Circuit_ID"].values
    compounds = np.round(compound.values).astype(int)  # encoded integers
    # Clamp between 0 and 4 (or max value in your map)
    compounds_clamped = np.clip(compounds, 0, 4)

    # Build a small DataFrame for easy grouping
    df = pd.DataFrame({
        "Race": race_ids,
        "Compound": compounds_clamped
    })

    # Process each race separately
    for race_id, group in df.groupby("Race"):
        comp = group["Compound"].values
        
        strategy = []
        current_compound = REVERSE_COMPOUND_MAP[comp[0]]
        stint_length = 1
        
        for i in range(1, len(comp)):
            if REVERSE_COMPOUND_MAP[comp[i]] == current_compound:
                stint_length += 1
            else:
                strategy.append((current_compound, stint_length))
                current_compound = REVERSE_COMPOUND_MAP[comp[i]]
                stint_length = 1
        
        # Add the last stint
        strategy.append((current_compound, stint_length))
        
        results[race_id] = strategy

    return results