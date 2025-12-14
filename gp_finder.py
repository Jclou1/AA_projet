import fastf1
import pandas as pd
import itertools

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
    )

    return events_df, unique_events

# Load all circuit from start_date to end_date
# Check if Verstappen driver is present
# Check if Verstappen has DNF
# Check if there was rain
# Output all circuits where Verstappen participated and did not DNF in dry conditions


def filter_circuits_verstappen_no_dnf_rain(start_year=2018, end_year=2024):
    circuits_names = ['Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China', 'Miami', 'Imola', 'Monaco', 'Canada', 'Spain', 'Austria', 'UK',
                      'Hungary', 'Belgium', 'Netherlands', 'Italy', 'Azerbaijan', 'Singapore', 'USA', 'Mexico', 'Brazil', 'Las Vegas', 'Qatar', 'Abu Dhabi']
    circuits = list(itertools.product(
        range(start_year, end_year + 1), circuits_names))
    drivers = ['VER', 'LEC', 'HAM', 'GAS', 'RUS', 'NOR']

    filtered_circuits = []

    for year, gp in circuits:
        print(f"Checking {year} {gp}...")
        try:
            session = fastf1.get_session(year, gp, 'R')
            session.load()
        except Exception as e:
            print(f"Could not load session for {year} {gp}: {e}")
            continue

        # Check if Verstappen participated
        laps = session.laps
        # drivers = laps['Driver'].unique()
        # if 'VER' not in drivers:
        #     continue

        # Check if nb of laps completed by all drivers is less than total laps (DNF)
        total_laps = session.total_laps
        has_dnf = False
        for driver in drivers:
            driver_laps = laps.pick_driver(driver)
            if not driver_laps.empty:
                if driver_laps['LapNumber'].max() < total_laps:
                    has_dnf = True
                    print(f"{driver} DNF in {year} {gp}")
                    break
        if has_dnf:
            continue

        # Check for rain during the race
        weather_data = laps.get_weather_data()
        if not weather_data.empty:
            if weather_data['Rainfall'].max() > 0:
                continue
            filtered_circuits.append((year, gp))

    return filtered_circuits


# all_events, all_circuits = list_all_events(2018, 2024)

# print("=== Circuits uniques FastF1 (2018-2024) ===")
# print(all_circuits)


filtered = filter_circuits_verstappen_no_dnf_rain(2018, 2024)
print("=== Circuits où Verstappen a participé, n'a pas DNF, et sans pluie (2018-2024) ===")
for year, gp in filtered:
    print(f"{year} - {gp}")
