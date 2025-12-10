import fastf1
import pandas as pd

fastf1.Cache.enable_cache(".fastf1_cache")

def list_all_events(start_year=2018, end_year=2024):
    events = []

    for year in range(start_year, end_year + 1):
        schedule = fastf1.get_event_schedule(year)
        # On garde quelques colonnes utiles
        df = schedule[["EventName", "Country", "Location"]].copy()
        df["Year"] = year
        events.append(df)

    events_df = pd.concat(events, ignore_index=True)

    # Circuits uniques par EventName
    unique_events = (
        events_df[["EventName", "Country", "Location"]]
        .drop_duplicates()
        .sort_values(["Country", "Location", "EventName"])
    )

    return events_df, unique_events

all_events, all_circuits = list_all_events(2018, 2024)

print("=== Circuits uniques FastF1 (2018–2024) ===")
print(all_circuits)
