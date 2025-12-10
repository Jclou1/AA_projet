import numpy as np
import pandas as pd
import itertools

COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

def simulate_strategy(
    model,
    strategy,
    total_laps,
    circuit_id,
    track_temp,
    pit_time
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

def generate_strategies(total_laps=57):
    compounds = ["SOFT", "MEDIUM", "HARD"]
    max_stops = 3  # jusqu'à 3 stops → 4 relais
    all_strategies = {}

    for n_stops in range(1, max_stops + 1):
        n_stints = n_stops + 1

        for comp_combo in itertools.product(compounds, repeat=n_stints):
            base_len = total_laps // n_stints
            stint_lengths = [base_len] * n_stints
            stint_lengths[-1] += total_laps - sum(stint_lengths)

            strategy = list(zip(comp_combo, stint_lengths))

            name = f"{n_stops} stops " + " → ".join(comp_combo)
            all_strategies[name] = strategy

    return all_strategies

