"""vB31 — 7e jeu, autre domaine : systèmes photovoltaïques (NREL PVDAQ), règles v1 (DB-09) et v2 (DB-10) gelées.

Le domaine retire les conditions par un modèle par défaut (ratio de performance, RdTools) ; la dégradation y est
lente (~0,5–1 %/an) et noyée dans les variations d'irradiance : R̂ attendu bas, premier test hors échantillon de la
branche « modèle » de v2.

Données : `vB31_pvdaq_charger.py` (nettoyage déclaré avant tout calcul contre l'âge). Exclusions, critère physique
fixé avant de mesurer, sans regarder l'âge : T_mod − T_amb médian à POA > 800 hors de [5, 35] °C, ou moins de 3 ans
de jours valides → 51 (−0,8 °C : capteur de module faux) et 1277 (398 jours ; T_amb de jour à 3 °C, invraisemblable).
Restent Las Vegas 34, 35, 1276, 1278 et Golden 4, 10, 33, 50.

Structure, déclarée AVANT tout calcul contre l'âge :
  - unité = système ; « vol » = jour ; fenêtre = 15 min de jour (POA > 200 W/m²) ;
  - S = [P_ac, P_dc] rapportées à la puissance installée, T_mod ; W = [POA, T_amb] ;
  - vérité = âge (jours depuis le premier jour valide du système) ; performance = Spearman(score moyen du jour,
    âge) par système de test, médiane sur les systèmes de test ;
  - usage non vu = climat non vu : entraînement sur 3 des 4 systèmes d'un site (4 rotations), test sur les 4
    systèmes de l'autre site ; dans les deux sens → 8 cas (ceux du critère) ;
  - contrôle rapporté à part : même site, 3 systèmes en entraînement, le 4e en test → 8 cas.
Méthodes, indicateurs : DB-09 (vB27). Règles : v1 (DB-09), v2 (DB-10).

Lecture fixée AVANT de mesurer : chaque règle est validée si, sur les 8 cas inter-sites, son regret médian est
≤ 0,05 et si aucune stratégie fixe n'a un regret médian plus bas de plus de 0,05. Rapportés aussi : R̂ par
entraînement (attendu < 1 si l'hypothèse « dégradation lente et faible » tient), cas à regret > 0,2.

Machine : Mac (Martin, 5 oct.). Sortie : voie_B_avec_exo/results/vB31_*.csv, vB31.txt
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
import vB30_capteurs_gaz as g30  # noqa: E402
import vB31_pvdaq_charger as ch  # noqa: E402

S_P = ["P_ac", "P_dc", "T_mod"]
W_P = ["POA", "T_amb"]
SITES = {"Las_Vegas": [34, 35, 1276, 1278], "Golden": [4, 10, 33, 50]}
M = r27.M


def cas(df, u_tr, u_te, cadre):
    tr = df[df.unit.isin(u_tr)]
    te = df[df.unit.isin(u_te)].reset_index(drop=True)
    S, W, un, cy = tr[S_P].values, tr[W_P].values, tr.unit.values, tr.cycle.values
    ind = r24.indicateurs(S, W, un, cy)
    perf = {}
    for nom, f in r27.methodes(S, W, un, cy).items():
        j = te.assign(s=f(te[S_P].values, te[W_P].values)).groupby(["unit", "age"]).s.mean().reset_index()
        perf[nom] = float(np.median([spearmanr(g.age, g.s).statistic for _, g in j.groupby("unit")]))
    return dict(cadre=cadre, entr=",".join(map(str, u_tr)), test=",".join(map(str, u_te)), **ind, **perf)


def main():
    df = ch.charger()
    df = df[df.unit.isin(sum(SITES.values(), []))].copy()
    df["age"] = (df.jour - df.groupby("unit").jour.transform("min")).dt.days
    rows = []
    for a, b in [("Las_Vegas", "Golden"), ("Golden", "Las_Vegas")]:
        for out in SITES[a]:
            u_tr = [u for u in SITES[a] if u != out]
            rows.append(cas(df, u_tr, SITES[b], f"inter-sites {a}→{b}"))
            rows.append(cas(df, u_tr, [out], f"contrôle {a}"))
    t = pd.DataFrame(rows)
    t["meilleure"] = t[M].max(axis=1)
    t["choix_v1"] = [r24.regle(p, D) for p, D in zip(t.p_hors, t.D)]
    t["choix_v2"] = [g30.regle_v2(R) for R in t.R_chapeau]
    for c in ["v1", "v2"]:
        t[f"regret_{c}"] = t.meilleure - t.apply(lambda r: r[r[f"choix_{c}"]], axis=1)
    for m in M:
        t[f"regret_{m}"] = t.meilleure - t[m]
    t.to_csv(v.OUT / "vB31_pvdaq.csv", index=False)

    pd.set_option("display.width", 250)
    print(df.groupby(["site", "unit"]).agg(jours=("jour", "nunique"), age_max=("age", "max")).to_string())
    print()
    print(t[["cadre", "entr", "test", "p_hors", "R_chapeau", "D", "lenteur", "modele", "span_W", "choix_v1",
             "choix_v2", "regret_v1", "regret_v2"]].round(2).to_string(index=False))
    R = ["regret_v1", "regret_v2"] + [f"regret_{m}" for m in M]
    inter = t[t.cadre.str.startswith("inter")]
    print("\n=== regret médian : inter-sites (critère) / contrôle ===")
    print(pd.DataFrame({"inter-sites": inter[R].median(), "contrôle": t[~t.index.isin(inter.index)][R].median()})
          .round(3).to_string())
    med = inter[R].median()
    fixes = med[[f"regret_{m}" for m in M]].min()
    for c in ["v1", "v2"]:
        ok = med[f"regret_{c}"] <= 0.05 and fixes >= med[f"regret_{c}"] - 0.05
        print(f"règle {c} validée sur ce jeu : {'OUI' if ok else 'NON'}")
    x = inter[inter.regret_v2 > 0.2]
    print("\ncas inter-sites à regret > 0,2 (v2) :")
    print(x[["cadre", "entr", "R_chapeau", "lenteur", "modele", "span_W", "choix_v2", "regret_v2"]].round(2)
          .to_string(index=False) if len(x) else "aucun")


if __name__ == "__main__":
    main()
