#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastF1 – Analyse F1 : extraction, EDA, visuels, modèle simple et rapport HTML.
Répond aux objectifs: extraction historique, visuels comparatifs, analyse de régularité,
et rapport synthèse (voir proposition de projet).  © Équipe 6
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import fastf1
from fastf1.core import Laps

# ---------- Préférences graphiques ----------
sns.set_context("talk")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (11, 6)
plt.rcParams["savefig.dpi"] = 120


# ---------- Utilitaires ----------
def ensure_dirs(out_dir: Path):
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)


def enable_cache(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def to_sec(td) -> float:
    """Convertit les LapTime (Timedelta) en secondes (float)."""
    if pd.isna(td):
        return np.nan
    return getattr(td, "total_seconds", lambda: float(td))()


# ---------- Chargement & préparation ----------
def load_session(year: int, gp: str | int, kind: str = "R"):
    """
    year:  ex. 2024
    gp:    nom officiel ('Bahrain', 'Monaco', 'Canada', ...) ou numéro de round (int)
    kind:  'R' (race), 'Q' (qualif), 'FP1'... (par défaut 'R')
    """
    ses = fastf1.get_session(year, gp, kind)
    ses.load()  # télécharge/parse télémétrie (utilise le cache si présent)
    return ses


def base_laps_df(ses) -> pd.DataFrame:
    """Retourne les tours course nettoyés avec colonnes utiles (robuste aux noms de colonnes)."""
    laps: Laps = ses.laps

    # Tours valides
    df = laps.copy()
    df = df[df["LapTime"].notna()].copy()
    if "IsAccurate" in df.columns:
        df = df[df["IsAccurate"]]

    # Colonnes de base
    df["LapTimeSec"] = df["LapTime"].apply(lambda td: td.total_seconds() if pd.notna(td) else np.nan)
    df["LapNumber"] = df["LapNumber"].astype(int)
    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].astype("category")
    if "Stint" in df.columns:
        df["Stint"] = df["Stint"].astype(int)
    if "TyreLife" in df.columns:
        df["TyreLife"] = df["TyreLife"].astype(int)

    # --- Clé temporelle pour joindre la météo ---
    # Priorité à LapStartTime (présent sur FastF1 >= 3.x). Sinon, on retombe sur Time.
    if "LapStartTime" in df.columns:
        df["_join_time"] = pd.to_timedelta(df["LapStartTime"])
    elif "Time" in df.columns:
        df["_join_time"] = pd.to_timedelta(df["Time"])
    else:
        # Secours : si rien n’existe, on crée un proxy croissant (moins précis mais évite le crash)
        df["_join_time"] = pd.to_timedelta(df["LapNumber"], unit="s")

    # --- Jointure météo (si disponible) ---
    if ses.weather_data is not None and not ses.weather_data.empty:
        wd = ses.weather_data.reset_index()  # l’index est 'Time'
        # Harmoniser en Timedelta et renommer la clé
        wd["_join_time"] = pd.to_timedelta(wd["Time"])
        wd = wd.sort_values("_join_time")

        df = pd.merge_asof(
            df.sort_values("_join_time"),
            wd[["_join_time", "TrackTemp", "AirTemp", "Humidity", "WindSpeed"]],
            on="_join_time",
            direction="nearest"
        )
    else:
        # Colonnes vides si pas de météo
        for c in ["TrackTemp", "AirTemp", "Humidity", "WindSpeed"]:
            if c not in df.columns:
                df[c] = np.nan

    return df.drop(columns=["_join_time"], errors="ignore")



# ---------- Analyses ----------
def driver_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Statistiques de régularité par pilote (tours de course uniquement)."""
    # Retirer SafetyCar/VirtualSC si la colonne existe
    if "TrackStatus" in df.columns:
        df = df[~df["TrackStatus"].astype(str).str.contains("4|5|6", na=False)]

    agg = (
        df.groupby("Driver")
        .agg(
            laps=("LapNumber", "count"),
            mean_laptime=("LapTimeSec", "mean"),
            std_laptime=("LapTimeSec", "std"),
            p25=("LapTimeSec", lambda s: np.nanpercentile(s, 25)),
            p50=("LapTimeSec", "median"),
            p75=("LapTimeSec", lambda s: np.nanpercentile(s, 75)),
            n_pitstops=("PitOutTime", "count"),
        )
        .reset_index()
    )
    agg["iqr"] = agg["p75"] - agg["p25"]
    agg["consistency_index"] = agg["std_laptime"] / agg["mean_laptime"]
    return agg.sort_values("mean_laptime")


def stint_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Résumé par pilote × relais (stint)."""
    cols = [
        "Driver",
        "Stint",
        "Compound",
        "TyreLife",
        "LapTimeSec",
        "LapNumber",
        "PitOutTime",
        "PitInTime",
    ]
    tmp = df[cols].copy()
    summ = (
        tmp.groupby(["Driver", "Stint", "Compound"])
        .agg(
            n_laps=("LapNumber", "count"),
            avg_pace=("LapTimeSec", "mean"),
            median_pace=("LapTimeSec", "median"),
            best_pace=("LapTimeSec", "min"),
            tyre_life_start=("TyreLife", "min"),
            tyre_life_end=("TyreLife", "max"),
            pitstop=("PitInTime", lambda s: int(s.notna().any())),
        )
        .reset_index()
    )
    return summ


# ---------- Visuels ----------
def plot_lap_times(df: pd.DataFrame, drivers: list[str], out_dir: Path) -> str:
    fig_path = out_dir / "figs" / "lap_times.png"
    plt.figure()
    for d in drivers:
        ddf = df[df["Driver"] == d]
        plt.plot(ddf["LapNumber"], ddf["LapTimeSec"], marker="o", ms=3, lw=1, label=d)
    plt.xlabel("Lap")
    plt.ylabel("Lap time (s)")
    plt.title("Lap time vs lap number")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return str(fig_path)


def plot_compound_box(df: pd.DataFrame, drivers: list[str], out_dir: Path) -> str:
    fig_path = out_dir / "figs" / "compound_box.png"
    plt.figure()
    sns.boxplot(
        data=df[df["Driver"].isin(drivers)],
        x="Compound",
        y="LapTimeSec",
        hue="Driver",
    )
    plt.title("Lap time distribution by compound")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return str(fig_path)


def plot_pace_delta_to_best(df: pd.DataFrame, drivers: list[str], out_dir: Path) -> str:
    """Δ-pace à la meilleure médiane du plateau (plus bas = mieux)."""
    med = df.groupby("Driver")["LapTimeSec"].median()
    ref = med.min()
    delta = (med - ref).sort_values()
    fig_path = out_dir / "figs" / "delta_to_best.png"
    plt.figure()
    delta.plot(kind="barh")
    plt.xlabel("Δ median lap time (s) vs best driver")
    plt.title("Pace delta to the best (median)")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return str(fig_path)


# ---------- Modèle prédictif simple ----------
def fit_predict_model(df: pd.DataFrame, drivers: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Modèle très simple: LapTime ~ TyreLife + LapNumber + Compound(OneHot) + TrackTemp.
    Entraîné sur le plateau, on donne le R² global et par pilote.
    """
    work = df.copy()
    # Sélection features
    features = ["TyreLife", "LapNumber", "TrackTemp", "Compound"]
    for c in features:
        if c not in work.columns:
            work[c] = np.nan

    work = work.dropna(subset=["LapTimeSec"])
    X = pd.get_dummies(work[features], columns=["Compound"], drop_first=True)
    y = work["LapTimeSec"].values

    mdl = LinearRegression()
    mdl.fit(X, y)
    work["Pred"] = mdl.predict(X)
    work["Err"] = work["Pred"] - work["LapTimeSec"]

    # Scores
    r2_global = r2_score(y, work["Pred"])
    r2_by_driver = (
        work.groupby("Driver")
        .apply(lambda g: r2_score(g["LapTimeSec"], g["Pred"]) if len(g) > 5 else np.nan)
        .to_dict()
    )

    info = {
        "r2_global": float(r2_global),
        "r2_by_driver": {k: (None if pd.isna(v) else float(v)) for k, v in r2_by_driver.items()},
        "coef_": dict(zip(X.columns, mdl.coef_.round(6))),
        "intercept_": float(mdl.intercept_),
    }

    # Exemple de table d’erreurs pour pilotes d’intérêt
    err_tbl = (
        work[work["Driver"].isin(drivers)]
        .groupby("Driver")
        .agg(n=("Err", "size"), mae=("Err", lambda s: float(np.abs(s).mean())), r2=("Err", lambda s: None))
        .reset_index()
    )
    # on ajoute R² pilote
    err_tbl["r2"] = err_tbl["Driver"].map(info["r2_by_driver"])
    return err_tbl, info


# ---------- Rapport ----------
REPORT_TMPL = """<!DOCTYPE html>
<html lang="fr"><head>
<meta charset="utf-8"><title>Rapport FastF1 – {{ title }}</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px}
h1,h2{margin:0 0 8px 0} h2{margin-top:24px}
table{border-collapse:collapse;margin:12px 0;width:100%}
th,td{border:1px solid #ddd;padding:6px;text-align:center}
.small{color:#666;font-size:0.9em}
img{max-width:100%;border:1px solid #eee}
code{background:#f6f8fa;padding:2px 4px;border-radius:4px}
</style>
</head><body>
<h1>Rapport d'analyse FastF1 – {{ title }}</h1>
<p class="small">Généré le {{ now }}</p>

<h2>Résumé</h2>
<ul>
  <li>Session: <b>{{ meta.kind }}</b>, Grand Prix: <b>{{ meta.name }}</b>, {{ meta.year }}</li>
  <li>Drivers sélectionnés: {{ drivers|join(", ") }}</li>
  <li>Tours considérés (après nettoyage): {{ n_laps }}</li>
</ul>

<h2>Régularité par pilote</h2>
{{ consistency_html }}

<h2>Résumé des relais (stints)</h2>
{{ stint_html }}

<h2>Δ-pace à la meilleure médiane du plateau</h2>
<img src="{{ figs.delta }}" alt="Pace delta">

<h2>Lap time vs lap number</h2>
<img src="{{ figs.laptimes }}" alt="Lap times">

<h2>Distribution par type de pneus</h2>
<img src="{{ figs.box }}" alt="Compound boxplot">

<h2>Modèle prédictif simple</h2>
<p>Régression linéaire: <code>LapTime ~ TyreLife + LapNumber + Compound(one-hot) + TrackTemp</code></p>
<ul>
  <li>R² global: <b>{{ model.r2_global|round(3) }}</b></li>
  <li>Coefs: <code>{{ model.coef_ }}</code>, intercept: <code>{{ model.intercept_|round(3) }}</code></li>
</ul>
{{ err_html }}

<p class="small">NB: Les métriques sont calculées sur les tours « propres » (pas de tours d'entrée/sortie stands). Les
conditions de piste (SC/VSC) sont écartées si disponibles.</p>
</body></html>
"""


def render_report(
    out_dir: Path,
    title: str,
    meta: dict,
    drivers: list[str],
    n_laps: int,
    consistency_df: pd.DataFrame,
    stint_df: pd.DataFrame,
    figs: dict,
    err_tbl: pd.DataFrame,
    model_info: dict,
) -> str:
    html_path = out_dir / "report.html"
    tpl = Template(REPORT_TMPL)

    html = tpl.render(
        title=title,
        now=datetime.now().strftime("%Y-%m-%d %H:%M"),
        meta=meta,
        drivers=drivers,
        n_laps=n_laps,
        consistency_html=consistency_df.to_html(index=False),
        stint_html=stint_df.to_html(index=False),
        figs={
            "laptimes": os.path.relpath(figs["laptimes"], out_dir),
            "box": os.path.relpath(figs["box"], out_dir),
            "delta": os.path.relpath(figs["delta"], out_dir),
        },
        err_html=err_tbl.to_html(index=False),
        model=model_info,
    )
    html_path.write_text(html, encoding="utf-8")
    return str(html_path)


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Analyse FastF1 et rapport HTML")
    p.add_argument("--year", type=int, default=2024, help="Année (ex: 2024)")
    p.add_argument("--gp", type=str, default="Bahrain", help="Nom GP (ex: Bahrain) ou round (int)")
    p.add_argument("--kind", type=str, default="R", choices=["R", "Q", "S", "FP1", "FP2", "FP3"],
                   help="Type de session (R=Race, Q=Qualif, S=Shootout/Sprint...)")
    p.add_argument("--drivers", type=str, default="VER,LEC",
                   help="Liste d'abréviations pilotes séparées par des virgules (ex: VER,PER,LEC)")
    p.add_argument("--out", type=str, default="outputs", help="Dossier de sortie")
    p.add_argument("--cache", type=str, default=".fastf1_cache", help="Dossier de cache FastF1")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    ensure_dirs(out_dir)
    enable_cache(Path(args.cache))

    # 1) Chargement
    gp = int(args.gp) if args.gp.isdigit() else args.gp
    ses = load_session(args.year, gp, args.kind)
    df = base_laps_df(ses)

    # 2) Analyses
    consistency = driver_consistency(df)
    stints = stint_summary(df)

    # 3) Visuels (pilotes d’intérêt)
    drivers = [d.strip().upper() for d in args.drivers.split(",") if d.strip()]
    figs = {
        "laptimes": plot_lap_times(df, drivers, out_dir),
        "box": plot_compound_box(df, drivers, out_dir),
        "delta": plot_pace_delta_to_best(df, drivers, out_dir),
    }

    # 4) Petit modèle
    err_tbl, model_info = fit_predict_model(df, drivers)

    # 5) Rapport
    meta = {"kind": args.kind, "name": ses.event["EventName"], "year": args.year}
    report_path = render_report(
        out_dir=out_dir,
        title=f"{args.year} – {ses.event['EventName']} ({args.kind})",
        meta=meta,
        drivers=drivers,
        n_laps=len(df),
        consistency_df=consistency,
        stint_df=stints,
        figs=figs,
        err_tbl=err_tbl,
        model_info=model_info,
    )
    print(f"\n✅ Rapport généré : {report_path}")
    print(f"   Figures dans : {out_dir/'figs'}")
    print(f"   Données (si vous en sauvegardez) : {out_dir/'data'}")


if __name__ == "__main__":
    main()
