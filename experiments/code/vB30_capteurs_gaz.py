"""vB30 — 6e jeu, autre domaine : dérive d'un réseau de capteurs de gaz (UCI 270), règles v1 (DB-09) et v2 (DB-10) gelées.

Données : `../../../Datasets/Gas_Sensor_Drift_Concentrations/` (UCI 270, CC BY 4.0, téléchargé le 5 oct. avec
l'accord de Martin ; citer Vergara et al. 2012, Sens. Actuators B ; Rodriguez-Lujan et al. 2014). 13 910 mesures
sur 36 mois, 16 capteurs à oxyde métallique × 8 caractéristiques, 6 gaz à concentrations variées. La dérive des
capteurs (vieillissement, empoisonnement) est la « dégradation » ; la concentration est la condition d'usage.

Constats sans étiquettes avant de déclarer (D5) : pas d'horodatage ; l'ordre des lignes à l'intérieur d'un lot
n'est pas chronologique (lot 10 : le gaz alterne à chaque ligne) ; les capteurs vont par paires d'échelle voisine
(1–2, 3–4, …), vraisemblablement de même type.

Structure, déclarée AVANT tout calcul contre la vérité (DB-09, DB-10) :
  - unité = capteur (16) ; « vol » = une mesure, ordonnée par (lot, ligne) — à l'intérieur d'un lot l'ordre est
    arbitraire (au plus quelques mois de dérive ignorés) ; une fenêtre par vol ;
  - S = les 8 caractéristiques du capteur (ΔR, ΔR normalisé, 3 moyennes mobiles montantes, 3 descendantes) ;
  - W = log10(concentration) ; le gaz n'est pas dans W : on traite chaque gaz séparément ;
  - vérité = numéro du lot (1–10, ordre des mois) ; performance = Spearman(score, lot) par capteur de test,
    médiane sur les capteurs de test ;
  - usage non vu : par gaz, entraînement sur les mesures de concentration ≤ médiane du gaz, test sur les mesures
    > médiane (hors support en W) ;
  - découpages des capteurs : impairs en entraînement / pairs en test, et l'inverse (un capteur de chaque paire
    de chaque côté). 6 gaz × 2 découpages = 12 cas.
Méthodes et indicateurs : ceux de DB-09 (vB27). Règles : v1 (DB-09) et v2 (DB-10 : R̂ < 1 → modèle, sinon lenteur).

Lecture fixée AVANT de mesurer (DB-09, DB-10) : chaque règle est validée sur ce jeu si son regret médian sur les
12 cas est ≤ 0,05 et si aucune stratégie fixe n'a un regret médian plus bas de plus de 0,05. v2 n'avait jamais vu
ce jeu : c'est sa première validation hors échantillon.

Machine : Mac (Martin, 5 oct.). Sortie : voie_B_avec_exo/results/vB30_*.csv, vB30.txt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import vB01_sonde_marge as v  # noqa: E402
import vB24_regle_choix as r24  # noqa: E402
import vB27_robustesse_decoupages as r27  # noqa: E402

DATA_ROOT = Path(os.environ.get("DATA_ROOT", ICI.parents[1] / "data"))  # données publiques, voir README
RACINE = DATA_ROOT / "Gas_Sensor_Drift_Concentrations"
S_G = [f"s{k}" for k in range(8)]
M = r27.M
R_SEUIL = 1.0


def regle_v2(R):
    return "modele" if R < R_SEUIL else "lenteur"


def charger() -> pd.DataFrame:
    rows = []
    for b in range(1, 11):
        for i, ligne in enumerate(open(RACINE / f"batch{b}.dat")):
            p = ligne.split()
            g, c = p[0].split(";")
            rows.append([b, i, int(g), float(c)] + [float(x.split(":")[1]) for x in p[1:]])
    d = pd.DataFrame(rows, columns=["lot", "ligne", "gaz", "conc"] + list(range(128)))
    d["cycle"] = np.arange(len(d))                       # ordre (lot, ligne)
    parts = []
    for k in range(16):                                  # une unité par capteur
        x = d[["lot", "cycle", "gaz", "conc"]].copy()
        x[S_G] = d[list(range(8 * k, 8 * k + 8))].values
        x["unit"] = k + 1
        parts.append(x)
    df = pd.concat(parts, ignore_index=True)
    df["logc"] = np.log10(df.conc)
    return df


def main():
    df = charger()
    rows = []
    for gaz in sorted(df.gaz.unique()):
        dg = df[df.gaz == gaz]
        med = dg.conc.median()
        for nom_dec, impairs_en_entr in [("impairs→pairs", True), ("pairs→impairs", False)]:
            u_tr = [u for u in range(1, 17) if (u % 2 == 1) == impairs_en_entr]
            tr = dg[dg.unit.isin(u_tr) & (dg.conc <= med)]
            te = dg[~dg.unit.isin(u_tr) & (dg.conc > med)].reset_index(drop=True)
            S, W, un, cy = tr[S_G].values, tr[["logc"]].values, tr.unit.values, tr.cycle.values
            ind = r24.indicateurs(S, W, un, cy)
            perf = {}
            for nom, f in r27.methodes(S, W, un, cy).items():
                t_ = te.assign(s=f(te[S_G].values, te[["logc"]].values))
                perf[nom] = float(np.median([spearmanr(g.lot, g.s).statistic for _, g in t_.groupby("unit")]))
            rows.append(dict(gaz=gaz, decoupage=nom_dec, n_tr=len(tr), n_te=len(te),
                             lots_te=te.lot.nunique(), **ind, **perf))
    t = pd.DataFrame(rows)
    t["meilleure"] = t[M].max(axis=1)
    t["choix_v1"] = [r24.regle(p, D) for p, D in zip(t.p_hors, t.D)]
    t["choix_v2"] = [regle_v2(R) for R in t.R_chapeau]
    for c in ["v1", "v2"]:
        t[f"regret_{c}"] = t.meilleure - t.apply(lambda r: r[r[f"choix_{c}"]], axis=1)
    for m in M:
        t[f"regret_{m}"] = t.meilleure - t[m]
    t.to_csv(v.OUT / "vB30_capteurs_gaz.csv", index=False)

    pd.set_option("display.width", 250)
    print(t[["gaz", "decoupage", "n_tr", "n_te", "lots_te", "p_hors", "R_chapeau", "D", "lenteur", "modele", "span_W",
             "choix_v1", "choix_v2", "regret_v1", "regret_v2"]].round(2).to_string(index=False))
    R = ["regret_v1", "regret_v2"] + [f"regret_{m}" for m in M]
    med = t[R].median()
    print("\n=== regret médian sur les 12 cas ===")
    print(med.round(3).to_string())
    fixes = med[[f"regret_{m}" for m in M]].min()
    for c in ["v1", "v2"]:
        ok = med[f"regret_{c}"] <= 0.05 and fixes >= med[f"regret_{c}"] - 0.05
        print(f"règle {c} validée sur ce jeu : {'OUI' if ok else 'NON'}")
    print("\ncas à regret > 0,2 (v2) :")
    x = t[t.regret_v2 > 0.2]
    print(x[["gaz", "decoupage", "R_chapeau", "lenteur", "modele", "span_W", "choix_v2", "regret_v2"]].round(2)
          .to_string(index=False) if len(x) else "aucun")


if __name__ == "__main__":
    main()
