"""vB29 — 5e jeu, jamais regardé : roulements de Paderborn sous vitesse et charges variables, règle v1 gelée (DB-09).

Données : `../../../Datasets/Bearings_UPB_TimeVarying/` (dépôt GitHub des auteurs, cloné le 5 oct. avec l'accord de
Martin ; caractéristiques dérivées de Zenodo 10868257, CC BY 4.0 ; citer Javanmardi, Aimiyekagbon et al., PHME 2024,
et le jeu Zenodo). 17 essais jusqu'à la panne ; entre deux mesures (~12 s), la vitesse et les charges sont tirées
au hasard dans une plage propre à l'essai. Pas de vérité de santé : l'âge (rang de la mesure) sert de proxy,
comme sur C-MAPSS et XJTU-SY.

Structure, déclarée AVANT tout calcul contre l'âge (DB-09) :
  - unité = roulement ; « vol » = une mesure (1,6 s de signal) ; une fenêtre par vol ;
  - capteurs S : les 200 caractéristiques des auteurs (maximum de l'amplitude FFT par bande égale, accéléromètre A)
    couvrent 0–64 kHz pour B01–B09 (échantillonnage 128 kHz) et 0–32 kHz pour B10–B17 (64 kHz) ; vérifié sans
    étiquettes : plancher de bruit au-delà de la bande ~120 pour B01–B09 seulement. Grille commune : 0–32 kHz en 20
    bandes de 1,6 kHz = maximum sur 5 bandes consécutives (B01–B09, bandes 0–99) ou sur 10 (B10–B17) ; le maximum
    d'une réunion de bandes est exact. S = log10 de ces 20 maxima + températures du palier T1, T2 (22 colonnes) ;
  - variables d'opération W : charge dynamique de crête, charge statique moyenne, vitesse moyenne (les trois
    retenues par les auteurs) et température ambiante ; une colonne est retirée d'un entraînement quand sa
    consigne y est constante (vitesse fixe de B14–B17, charges fixes de B11–B13…) ;
  - mesures écartées : rodage initial (consigne de vitesse < 1 000 tr/min, ~50 mesures de B01–B10) ;
  - B01 (aucun critère d'arrêt) et B05 (essai interrompu) : entraînement seulement, jamais en test.
Groupes d'usage (plages de conditions et excitation dynamique ; la forme de l'excitation n'est pas dans W) :
  - G1 sinus, trois conditions variables : B01–B06 ;
  - G2 bruit, trois conditions variables : B07–B10 ;
  - G3 bruit, vitesse seule variable : B11–B13 ;
  - G4 bruit, une charge seule variable, vitesse fixe : B14–B17.
Découpages : chaque groupe tour à tour en entraînement, test = chacun des 3 autres groupes → 12 cas.
Méthodes, indicateurs, règle : ceux de vB27 / DB-09, inchangés. Performance = Spearman(score, rang de la mesure)
par roulement de test, médiane sur les roulements du groupe de test.

Lecture fixée AVANT de mesurer (DB-09) : la règle est validée sur ce jeu si son regret médian sur les 12 cas est
≤ 0,05 et si aucune stratégie fixe n'a un regret médian plus bas de plus de 0,05 ; sinon, échec rapporté.
Aussi rapportés : p_hors, D par entraînement ; cas à regret > 0,2.

Machine : Mac (Martin, 5 oct.). Sortie : voie_B_avec_exo/results/vB29_*.csv, vB29.txt
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
RACINE = DATA_ROOT / "Bearings_UPB_TimeVarying/data/LDM"
S_U = [f"F{k:02d}" for k in range(20)] + ["T1", "T2"]
W_U = ["dyn", "stat", "vitesse", "T_amb"]
CONSIGNE = {"dyn": "setDynLoad / N", "stat": "setStatLoad / N", "vitesse": "setSpeed / rpm"}
GROUPES = {"G1_sinus": [1, 2, 3, 4, 5, 6], "G2_bruit_3var": [7, 8, 9, 10], "G3_vitesse": [11, 12, 13],
           "G4_charge": [14, 15, 16, 17]}
ENTR_SEULEMENT = {1, 5}
M = r27.M


def charger() -> pd.DataFrame:
    parts = []
    for i in range(1, 18):
        b = f"B{i:02d}"
        X = np.load(RACINE / f"fft/{b}.npy")
        op = pd.read_csv(RACINE / f"op/{b}.csv")
        T = pd.read_csv(RACINE / f"temp/{b}.csv")
        n = min(len(X), len(op), len(T))
        X, op, T = X[:n], op.iloc[:n].reset_index(drop=True), T.iloc[:n].reset_index(drop=True)
        k = 5 if i <= 9 else 10
        F = np.log10(X[:, :20 * k].reshape(n, 20, k).max(axis=2))
        d = pd.DataFrame(F, columns=S_U[:20])
        d["T1"], d["T2"] = T.iloc[:, 1].values, T.iloc[:, 2].values
        d["dyn"], d["stat"], d["vitesse"] = op["peak_dynLoad / N"], op["meanAbs_statLoad / N"], op["meanAbs_speed / rpm"]
        d["T_amb"] = T.iloc[:, 3].values
        for c, s in CONSIGNE.items():
            d[f"set_{c}"] = op[s].values
        d["unit"], d["cycle"] = i, np.arange(n)
        parts.append(d[op["setSpeed / rpm"].values >= 1000])
    df = pd.concat(parts, ignore_index=True)
    df["groupe"] = df.unit.map({u: g for g, us in GROUPES.items() for u in us})
    return df


def main():
    df = charger()
    rows = []
    for g_tr in GROUPES:
        tr = df[df.groupe == g_tr]
        Wk = [c for c in W_U if c == "T_amb" or tr[f"set_{c}"].nunique() > 1]
        S, W, un, cy = tr[S_U].values, tr[Wk].values, tr.unit.values, tr.cycle.values
        ind = r24.indicateurs(S, W, un, cy)
        fs = r27.methodes(S, W, un, cy)
        for g_te in GROUPES:
            if g_te == g_tr:
                continue
            te = df[(df.groupe == g_te) & ~df.unit.isin(ENTR_SEULEMENT)].reset_index(drop=True)
            perf = {}
            for nom, f in fs.items():
                t_ = te.assign(s=f(te[S_U].values, te[Wk].values))
                perf[nom] = float(np.median([spearmanr(g.cycle, g.s).statistic for _, g in t_.groupby("unit")]))
            rows.append(dict(entr=g_tr, test=g_te, W=",".join(Wk), **ind, **perf))
    t = pd.DataFrame(rows)
    t["meilleure"] = t[M].max(axis=1)
    t["choix"] = [r24.regle(p, D) for p, D in zip(t.p_hors, t.D)]
    t["regret_regle"] = t.meilleure - t.apply(lambda r: r[r.choix], axis=1)
    for m in M:
        t[f"regret_{m}"] = t.meilleure - t[m]
    t.to_csv(v.OUT / "vB29_roulements_upb.csv", index=False)

    pd.set_option("display.width", 250)
    print("=== mesures par roulement (après rodage) ===")
    print(df.groupby(["groupe", "unit"]).size().to_string())
    print()
    print(t[["entr", "test", "W", "p_hors", "D", "lenteur", "modele", "span_W", "choix", "regret_regle"]]
          .round(2).to_string(index=False))
    R = ["regret_regle"] + [f"regret_{m}" for m in M]
    med = t[R].median()
    print("\n=== regret médian sur les 12 cas ===")
    print(med.round(3).to_string())
    ok = med.regret_regle <= 0.05 and med[R[1:]].min() >= med.regret_regle - 0.05
    print(f"\nrègle validée sur ce jeu (DB-09) : {'OUI' if ok else 'NON'}")
    x = t[t.regret_regle > 0.2]
    print("\ncas à regret > 0,2 :")
    print(x[["entr", "test", "p_hors", "D", "lenteur", "modele", "span_W", "choix", "regret_regle"]].round(2)
          .to_string(index=False) if len(x) else "aucun")


if __name__ == "__main__":
    main()
