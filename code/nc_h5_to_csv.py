"""Convertit N-CMAPSS_DS01.h5 -> CSV au format attendu par H-CDE (EPFL-IMOS/hcde).

Colonnes produites (cf. src/dataset/cmapss.py de H-CDE) :
    alt, Mach, TRA, T2                       (w_vars)
    T24..Wf (14 capteurs)                    (x_vars)
    HPT_eff_mod                              (paramètre de santé vrai, ÉVAL seulement)
    unit, cycle, hs, RUL

RUL = dernier cycle de l'unité - cycle courant. hs et HPT_eff_mod sont la vérité terrain, jamais
en entrée du modèle (H-CDE ne lit que w_vars+x_vars ; y_label sert à l'évaluation).

Sous-échantillonnage : on garde 1 point sur `STRIDE` dans chaque (unité, cycle). C'est un choix de
coût, déclaré ; il ne change ni les valeurs ni l'ordre. Les unités retenues sont celles du config
H-CDE (train [1,4,7], test [9]) plus quelques-unes pour notre propre éval covariate.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import argparse

import os

# Chemins surchargeables par variables d'environnement (NCMAPSS_DIR, HCDE_DATASETS, DATA_ROOT ; voir README).
RACINE = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NCMAPSS_DIR",
                           Path(os.environ.get("DATA_ROOT", RACINE / "data")) / "N-CMAPSS"))
OUT = Path(os.environ.get("HCDE_DATASETS", RACINE / "hcde/run/datasets/ncmapss"))
# UNITS=None -> toutes les unités du DS (dev + test). Les configs H-CDE sélectionnent ensuite via
# train_units/test_units. CORRECTION (15 sept.) : le split officiel [1,4,7]->[9] du papier n'est PAS
# une « démo » — c'est bien leur setup, et ces unités sont toutes de classe de vol Fc=1 (vols courts),
# ce qui correspond à leur « short flights ». Chaque unité N-CMAPSS a UNE classe de vol ; DS01 couvre
# Fc 1,2,3 selon l'unité (u1,u4,u7,u9 = Fc1 ; u8 = Fc2 ; u10 = Fc3).
STRIDE = 10          # 0,1 Hz = recette du papier H-CDE (Chao et al.) ; fenêtre lente ~4-5 vols
W = ["alt", "Mach", "TRA", "T2"]
XS = ["T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24", "Ps30", "P40", "P50", "Nf", "Nc", "Wf"]


def main(ds: str = "DS01", units=None):
    global H5, UNITS
    H5 = DATA / f"N-CMAPSS_{ds}.h5"
    UNITS = units
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(H5, "r") as f:
        Tn = [x.decode() if isinstance(x, bytes) else x for x in np.array(f["T_var"]).ravel()]
        hpt = Tn.index("HPT_eff_mod")
        # CORRECTION (15 sept.) — BUG : n'exporter que HPT_eff_mod ne marche que pour DS01.
        # Chaque DS N-CMAPSS dégrade un composant DIFFÉRENT (DS04 fan, DS05 HPC, DS06 LPC,
        # DS07 LPT...) : HPT_eff_mod y vaut EXACTEMENT 0, donc la « vraie dégradation » était
        # une constante → corrélations NaN et R² = 1,000 trompeurs. On exporte donc TOUS les
        # paramètres de santé, plus une cible `degradation` = norme L2 des paramètres ACTIFS
        # (étendue > 1e-4 sur ce DS). Sur DS01 un seul est actif → se réduit à |HPT_eff_mod|,
        # donc identique au protocole du papier.
        Tdev = np.asarray(f["T_dev"], np.float32)
        rng_T = Tdev.max(0) - Tdev.min(0)
        actifs = [i for i in range(len(Tn)) if rng_T[i] > 1e-4]
        print(f"paramètres de santé actifs sur {ds} : {[Tn[i] for i in actifs]}")
        frames = []
        for part in ("dev", "test"):
            A = np.asarray(f[f"A_{part}"], np.float32)
            unit = A[:, 0].astype(int)
            keep = np.ones(len(unit), bool) if UNITS is None else np.isin(unit, UNITS)
            if not keep.any():
                continue
            d = pd.DataFrame(np.asarray(f[f"W_{part}"])[keep], columns=W)
            d[XS] = np.asarray(f[f"X_s_{part}"])[keep]
            Tp = np.asarray(f[f"T_{part}"], np.float32)[keep]
            d["HPT_eff_mod"] = Tp[:, hpt]          # conservé : le loader H-CDE lit ce nom
            for i in actifs:                        # tous les paramètres réellement actifs
                d[f"T_{Tn[i]}"] = Tp[:, i]
            # cible d'évaluation, valide sur TOUS les DS
            d["degradation"] = np.linalg.norm(Tp[:, actifs], axis=1)
            d["unit"] = unit[keep]
            d["cycle"] = A[keep, 1].astype(int)
            d["Fc"] = A[keep, 2].astype(int)     # classe de vol (constante par unité) — covariate
            d["hs"] = A[keep, 3].astype(int)
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # RUL par unité, puis sous-échantillonnage dans chaque (unit, cycle)
    last = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (last - df["cycle"]).astype(int)
    # sous-échantillonnage : garder 1 point sur STRIDE dans chaque (unit, cycle),
    # sans groupby.apply (qui, en pandas récent, retire les colonnes de groupe).
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    pos = df.groupby(["unit", "cycle"], group_keys=False).cumcount()
    df = df[pos % STRIDE == 0].reset_index(drop=True)
    df["time"] = df.groupby("unit").cumcount()

    out = OUT / f"N-CMAPSS_{ds}.csv"
    # écriture ATOMIQUE : des entraînements en cours peuvent lire ce fichier au démarrage.
    tmp = out.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    print(f"écrit {out}: {len(df):,} lignes, unités {sorted(df.unit.unique())}")
    print(f"  cible `degradation` : min={df.degradation.min():.5f} max={df.degradation.max():.5f}")
    print(df.groupby("unit").agg(n=("cycle", "size"), cycles=("cycle", "nunique"),
                                 Fc=("Fc", "first"), hs1=("hs", "mean")).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ds", default="DS01", help="DS01, DS02, ... (fichier N-CMAPSS_<ds>.h5)")
    a = p.parse_args()
    main(a.ds)
