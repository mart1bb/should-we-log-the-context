"""vB48 — Incertitude des résultats batteries et « quoi journaliser » (2e relecture, W4).

Aucune nouvelle mesure : relit vB39_cas, vB42_{ablation,frequence,granularite,c1}. Pour chaque contraste, écart
apparié par cas, médiane, IC à 95 % par bootstrap sur les cas (2 000 tirages, graine 0 ; les valeurs par unité ne
sont pas gardées par vB42, d'où un bootstrap à un niveau), test de Wilcoxon signé (bilatéral). Contrastes :
  - batteries : résidu quadratique − meilleure sans contexte (max des trois a priori temporels) ;
  - attributs (batteries) : meilleure des 3 méthodes linéaires avec {I}, {Ah}, {I, Ah} − meilleure sans contexte ;
  - fréquence : meilleure des 3 méthodes avec W par cycle − avec W par fenêtre (N-CMAPSS, batteries, PVDAQ) ;
  - granularité (batteries) : normalisation par régime K = 12, 24 − meilleure sans contexte ;
  - référence saine sous C1 : span_W_ref − lenteur, par scénario (12 découpages).
Sortie : results/vB48.txt, vB48_contrastes.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES = Path(__file__).resolve().parent.parent / "results"
B = 2000


def resume(x, nom):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(0)
    bs = np.median(x[rng.integers(0, len(x), (B, len(x)))], axis=1)
    p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    return dict(contraste=nom, n=len(x), mediane=np.median(x), ic_bas=np.percentile(bs, 2.5),
                ic_haut=np.percentile(bs, 97.5), p_wilcoxon=p)


def main():
    c = pd.read_csv(RES / "vB39_cas.csv")
    c["cf"] = c[["lenteur", "tendance_moy", "tendance_reg"]].max(axis=1)
    bat = c[c.jeu == "batteries"].set_index("cas")
    rows = [resume(bat.resid_quad - bat.cf, "batteries : résidu quadratique − meilleure sans contexte")]
    a = pd.read_csv(RES / "vB42_ablation.csv")
    a = a[a.jeu == "batteries"].pivot_table(index=["cas", "attributs"], columns="methode", values="rho").max(axis=1)
    for s in ["I", "Ah", "I+Ah"]:
        rows.append(resume(a.xs(s, level="attributs") - bat.cf.reindex(a.xs(s, level="attributs").index),
                           f"batteries : attributs {s} − meilleure sans contexte"))
    f = pd.read_csv(RES / "vB42_frequence.csv")
    f = f.pivot_table(index=["jeu", "cas", "frequence"], columns="methode", values="rho").max(axis=1).unstack()
    for j in ["N-CMAPSS", "batteries", "PVDAQ"]:
        rows.append(resume((f.loc[j, "vol"] - f.loc[j, "fenetre"]).values, f"{j} : W par cycle − W par fenêtre"))
    g = pd.read_csv(RES / "vB42_granularite.csv")
    g = g[g.jeu == "batteries"].pivot_table(index="cas", columns="methode", values="rho")
    for K in (12, 24):
        rows.append(resume(g[f"norm_regime_K{K}"] - bat.cf.reindex(g.index), f"batteries : régimes K = {K} − meilleure sans contexte"))
    r = pd.read_csv(RES / "vB42_c1.csv")
    for sc, x in r.groupby("scenario"):
        rows.append(resume(x.span_W_ref - x.lenteur, f"C1 {sc} : span_W_ref − lenteur"))
    T = pd.DataFrame(rows)
    T.to_csv(RES / "vB48_contrastes.csv", index=False)
    pd.set_option("display.width", 250)
    txt = T.round(3).to_string(index=False)
    print(txt)
    (RES / "vB48.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
