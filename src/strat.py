import numpy as np
import pandas as pd

COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

def simulate_strategy(
    model,
    strategy,
    total_laps,
    circuit_id,
    track_temp=35,
    pit_time=22
):
    """
    Simule une course complète avec une stratégie donnée.

    strategy = liste de tuples :
       [("SOFT", 15), ("MEDIUM", 20), ("HARD", 22)]
    signifie :
       15 tours en soft,
       puis pit,
       20 tours en medium,
       puis pit,
       22 tours en hard

    Retourne : temps total de course (secondes)
    """

    race_time = 0
    lap_number = 1
    tyre_life = 1

    for compound, stint_length in strategy:
        compound_code = COMPOUND_MAP[compound]

        for _ in range(stint_length):
            X = pd.DataFrame({
                "TyreLife": [tyre_life],
                "Compound_Encoded": [compound_code],
                "LapNumber": [lap_number],
                "Circuit_ID": [circuit_id],
                "TrackTemp": [track_temp]
            })

            lap_time = model.predict(X)[0]
            race_time += lap_time

            lap_number += 1
            tyre_life += 1

        # On ajoute le temps du pit stop sauf au dernier relais
        race_time += pit_time
        tyre_life = 1  # pneus neufs

    return race_time
