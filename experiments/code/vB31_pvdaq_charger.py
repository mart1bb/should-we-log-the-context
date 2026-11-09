"""vB31 (chargement) — PVDAQ : nettoyage déclaré AVANT tout calcul contre l'âge (DB-10), mise en cache.

Données : `../../../Datasets/PVDAQ/` (OEDI PVDAQ, NREL, BSD-3 ; téléchargé le 5 oct. avec l'accord de Martin).
Las Vegas (BWh) : 34, 35, 1276, 1277, 1278, toutes années ; Golden (BSk) : 4, 10, 33, 50, 51, 2010–2019.

Constats sans étiquettes (D5, 25 jours tirés au hasard par système) : canaux renommés selon les années (même nom,
identifiants différents) ; sentinelles (−7999, −99999) et capteurs morts (colonne constante sur la journée) ;
puissances de Las Vegas tantôt en W, tantôt en hW selon le fichier ; températures de Las Vegas en °F ; pas de 15 min
(Las Vegas) ou 1 min (Golden).

Nettoyage, fixé avant de regarder toute tendance :
  - colonnes regroupées par nom (identifiant retiré) ; doublons fusionnés en gardant la première valeur valide ;
  - valeur invalide : NaN, ≤ −999 ; une colonne constante sur les lignes de jour d'un fichier est invalide ce jour-là ;
  - grandeurs : POA = `poa_irradiance` ; T_amb = `ambient_temp*` ; T_mod = médiane des `module_temp*` valides ;
    P_ac = `ac_power*` (système, pas un onduleur ni un compteur), P_dc = `dc_power*` (idem) ;
  - unités : °F → °C pour les colonnes `_f` ; puissance : par fichier, facteur k ∈ {1 (W), 100 (hW), 1000 (kW)} tel
    que k × médiane(P / POA) sur POA > 600 soit le plus proche, en log, de 0,8 × puissance installée (kW) — les
    facteurs sont séparés de ×10 au moins, l'âge (< 10 % de perte) ne peut pas changer le choix ;
  - puissances rapportées à la puissance installée (kW / kW) ;
  - pas commun 15 min (moyenne) ; fenêtres de jour : POA > 200 W/m² ; « vol » = jour ; jours gardés s'ils ont au
    moins 8 fenêtres valides sur toutes les colonnes.
Sortie : `voie_B_avec_exo/cache/pvdaq_fenetres.pkl` (non versionné).
"""
from __future__ import annotations

import glob
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ICI = Path(__file__).resolve().parent
import os
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ICI.parents[1] / "data"))  # données publiques, voir README
RACINE = DATA_ROOT / "PVDAQ"
CACHE = ICI.parent / "cache/pvdaq_fenetres.pkl"
SYSTEMES = {34: ("Las_Vegas", 146.64), 35: ("Las_Vegas", 121.68), 1276: ("Las_Vegas", 68.48),
            1277: ("Las_Vegas", 40.56), 1278: ("Las_Vegas", 171.36),
            4: ("Golden", 1.0), 10: ("Golden", 1.12), 33: ("Golden", 2.4), 50: ("Golden", 6.0), 51: ("Golden", 6.0)}
COLS = ["POA", "T_amb", "T_mod", "P_ac", "P_dc"]


def coalesce(d: pd.DataFrame, motif: str) -> pd.Series | None:
    cs = [c for c in d.columns if re.fullmatch(motif, c)]
    if not cs:
        return None
    out = pd.Series(np.nan, index=d.index)
    for c in cs:
        out = out.fillna(d[c])
    return out


def lire_jour(chemin: str, cap: float) -> pd.DataFrame | None:
    d = pd.read_csv(chemin)
    if "measured_on" not in d:
        return None
    t = pd.to_datetime(d.pop("measured_on"), errors="coerce")
    d = d.drop(columns=[c for c in d.columns if c == "system_id"])
    d.columns = [re.sub(r"__\d+$", "", c) for c in d.columns]
    d = d.apply(pd.to_numeric, errors="coerce")
    d = d.mask(d <= -999)
    d = d.T.groupby(level=0).first().T if d.columns.duplicated().any() else d   # doublons de nom
    poa = coalesce(d, r"poa_irradiance")
    if poa is None:
        return None
    jour = poa > 200
    if jour.sum() < 8:
        return None
    for c in d.columns:                                   # capteur mort ce jour-là
        if d.loc[jour, c].nunique(dropna=True) <= 1:
            d[c] = np.nan
    poa = coalesce(d, r"poa_irradiance")
    x = pd.DataFrame(index=d.index)
    x["POA"] = poa
    ta = coalesce(d, r"ambient_temp(_f)?")
    if ta is not None and any(c.endswith("_f") for c in d.columns if c.startswith("ambient_temp")):
        ta = (ta - 32) / 1.8
    x["T_amb"] = ta
    mods = [c for c in d.columns if c.startswith("module_temp")]
    tm = d[mods].median(axis=1) if mods else None
    if tm is not None and any(c.endswith("_f") for c in mods):
        tm = (tm - 32) / 1.8
    x["T_mod"] = tm
    for nom, motif in [("P_ac", r"ac_power(_hw)?"), ("P_dc", r"dc_power(_hw)?")]:
        p = coalesce(d, motif)
        if p is None:
            x[nom] = np.nan
            continue
        sel = (poa > 600) & (p > 0)
        if sel.sum() < 3:
            x[nom] = np.nan
            continue
        r = np.median(p[sel] / poa[sel])
        k = min([1.0, 100.0, 1000.0], key=lambda k: abs(np.log10(r * k) - np.log10(0.8 * cap)))
        x[nom] = p * k / 1000 / cap                       # k : W par unité du fichier → kW / kW installé
    x.index = t
    x = x[x.index.notna()].resample("15min").mean()
    x = x[(x.POA > 200)].dropna()
    return x if len(x) >= 8 else None


def lire_systeme(s: int) -> pd.DataFrame:
    site, cap = SYSTEMES[s]
    parts = []
    for f in sorted(glob.glob(str(RACINE / f"system_id={s}/year=*/month=*/day=*/*.csv"))):
        try:
            x = lire_jour(f, cap)
        except Exception as e:                            # fichier illisible : compté, pas silencieux
            print(f"illisible {f}: {e}", file=sys.stderr)
            continue
        if x is not None:
            parts.append(x)
    df = pd.concat(parts).sort_index()
    df["unit"], df["site"] = s, site
    df["jour"] = df.index.normalize()
    return df.reset_index(names="t")


def charger() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    with Pool(10) as pool:
        df = pd.concat(pool.map(lire_systeme, list(SYSTEMES)), ignore_index=True)
    jours = sorted(df.jour.unique())
    df["cycle"] = df.jour.map({j: i for i, j in enumerate(jours)})
    CACHE.parent.mkdir(exist_ok=True)
    df.to_pickle(CACHE)
    return df


if __name__ == "__main__":
    d = charger()
    pd.set_option("display.width", 220)
    print(d.groupby(["site", "unit"]).agg(fenetres=("t", "size"), jours=("jour", "nunique"), debut=("jour", "min"),
                                          fin=("jour", "max"), P_ac=("P_ac", "median"), P_dc=("P_dc", "median"),
                                          T_mod=("T_mod", "median"), T_amb=("T_amb", "median"),
                                          POA=("POA", "median")).to_string(float_format="%.2f"))
    p = d[d.POA > 800]
    print("\nT_mod − T_amb à POA > 800 (attendu ~15–25 °C) :")
    print((p.T_mod - p.T_amb).groupby(p.unit).median().round(1).to_string())
