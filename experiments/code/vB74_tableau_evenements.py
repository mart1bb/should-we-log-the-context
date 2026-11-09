"""vB74 — Tableau IV du papier (journal d'événements, f = 1) refait avec les expériences des reviews (vB71).

Remplace le tableau de vB68 (vB68 garde la figure, CARE, le placebo et les alarmes). Valeurs comme vB68 : par cas,
médiane sur les 4 méthodes de (meilleure des deux estimations de la réponse − sans correction), les 2 tirages étant
moyennés ; par jeu et ligne : médiane sur les cas, Wilcoxon apparié, Holm sur toutes les cases du tableau.
Lignes : mesure ; opération linéaire (vB67) ; opération d'un seul côté (`prod_seuil`) ; opération quadratique
(`prod_quad`) ; maintenance. Colonnes : dégât ; dates détectées ; dates du journal ; + catégorie ; + capteurs retrouvés
dans les données (`cat_infere`) ; + capteurs nommés. Maintenance : « + catégorie » = vies d'entraînement coupées ;
« + capteurs nommés » = marche sur les capteurs nommés SANS couper (`capt_sans_couper`), car couper et retirer la marche
sur les capteurs nommés crée un artefact de la SFA (vB73) ; « + capteurs retrouvés » n'est pas évalué sans coupe (--).
Dates détectées des catégories d'origine : vB67 (mêmes cas, mêmes graines ; vB71 en reproduit les autres réponses).
Sorties : papier/tab_events.tex, results/vB74.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import vB68_partie2_papier as b68

RES, PAP = b68.RES, b68.PAP
METH = b68.METH
LIGNES = [("mesure", "measurement"), ("production", "operation, linear"), ("prod_seuil", "operation, one-sided"),
          ("prod_quad", "operation, quadratic"), ("maintenance", "maintenance")]
COLS = [("degat", "damage"), ("detection", "detected dates"), ("date", "logged dates"), ("cat", "+ category"),
        ("infere", "+ sensors from data"), ("nommes", "+ named sensors")]


def par_cas():
    R = pd.read_csv(RES / "vB71_cas.csv")
    R = R[(R["mode"].isin(["injecte", "-"])) & (R["eval"] == "global")]
    L = R.melt(id_vars=["jeu", "cas", "categorie", "tirage", "reponse", "estimation"], value_vars=METH,
               var_name="methode", value_name="rho")
    propre = L[L.reponse == "propre"].set_index(["jeu", "cas", "methode"]).rho
    L = L[L.reponse != "propre"]
    P = L.pivot_table(index=["jeu", "cas", "categorie", "methode"], columns=["reponse", "estimation"], values="rho",
                      aggfunc="mean")
    best = lambda r: P[[c for c in P.columns if c[0] == r]].max(axis=1)                  # noqa: E731
    sans = P[("sans_journal", "-")]
    V = pd.DataFrame(index=P.index)
    V["degat"] = sans - propre.reindex(P.index.droplevel("categorie")).values
    V["detection"] = best("detection") - sans
    V["date"] = best("date") - sans
    V["cat"] = best("cat") - sans
    V["infere"] = best("cat_infere") - sans
    V["nommes"] = best("cat_capteurs") - sans
    m = V.index.get_level_values("categorie") == "maintenance"
    V.loc[m, "nommes"] = (best("capt_sans_couper") - sans)[m]
    V.loc[m, "infere"] = np.nan
    V.loc[V.index.get_level_values("categorie") == "mesure", "cat"] = np.nan          # identique à la date
    V = V.groupby(level=["jeu", "cas", "categorie"]).median()                           # médiane sur les méthodes
    D = b68.valeurs_par_cas().xs(1.0, level="ampleur")["g_detection"]                   # vB67
    for c in ("mesure", "production", "maintenance"):
        idx = V.index[V.index.get_level_values("categorie") == c]
        V.loc[idx, "detection"] = D.reindex(idx).values
    return V


def main():
    V = par_cas()
    S = []
    for j, _ in b68.JEUX:
        for c, _ in LIGNES:
            for q, _ in COLS:
                x = V.xs((j, c), level=["jeu", "categorie"])[q].values
                if np.all(np.isnan(x)):
                    S.append(dict(jeu=j, cat=c, q=q, med=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=len(x)))
                    continue
                med, lo, hi, p, n = b68.resume(x)
                S.append(dict(jeu=j, cat=c, q=q, med=med, lo=lo, hi=hi, p=p, n=n))
    S = pd.DataFrame(S)
    S["p_holm"] = b68.holm(S.p.values)
    out = ["% Généré par code/vB74_tableau_evenements.py — ne pas éditer à la main.",
           r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
           r"Dataset & Change & Damage & \multicolumn{5}{c}{Gain over no correction, with} \\",
           r"\cmidrule(l){4-8}",
           r" & & & detected & logged & + category & + sensors & + named \\",
           r" & & & dates & dates & & from data & sensors \\", r"\midrule"]
    for j, jn in b68.JEUX:
        n = int(S[S.jeu == j].n.iloc[0])
        for i, (c, cn) in enumerate(LIGNES):
            cells = []
            for q, _ in COLS:
                r = S[(S.jeu == j) & (S.cat == c) & (S.q == q)].iloc[0]
                if np.isnan(r.med):
                    cells.append("--")
                else:
                    t = b68.fmt(r.med)
                    cells.append(rf"\textbf{{{t}}}" if r.p_holm < 0.05 else t)
            tete = f"{jn} ({n})" if i == 0 else ""
            out.append(f"{tete} & {cn} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule" if j != b68.JEUX[-1][0] else r"\bottomrule")
    out.append(r"\end{tabular}")
    (PAP / "tab_events.tex").write_text("\n".join(out) + "\n")
    txt = "Tableau IV (f = 1, médiane des méthodes ; IC, p de Holm sur le tableau)\n" + S.round(3).to_string(index=False)
    (RES / "vB74.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
