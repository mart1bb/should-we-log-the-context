"""vB72 — Batteries : la perte de la lenteur (C3) vient-elle de la température ambiante ? (reviews IA du 20 oct., R1-Q1)

Lecture descriptive des résultats par unité de vB39 (aucun nouveau calcul de méthode). Groupes de vB22 : uniforme,
biaisée basse et haute à température ambiante (TA), biaisée basse et haute à 40 °C. La température ambiante est connue
par groupe mais constante dans chaque groupe d'entraînement : un modèle de contexte ne peut pas l'apprendre.
Par paire (groupe d'entraînement → groupe de test), médiane des Spearman par unité ; puis médiane des paires selon que
les deux groupes ont la même température ambiante ou non.
Si la température explique la perte de la lenteur, celle-ci doit être nettement meilleure à température identique.
Les paires à même température étant surtout uniforme ↔ biaisée, on croise aussi le profil de courant (même profil,
basse ↔ haute, uniforme ↔ biaisée) avec le changement de température.

Sortie : results/vB72.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
METH = ["lenteur", "tendance_reg", "span_W", "resid_lin", "resid_quad"]


def main():
    R = pd.read_csv(RES / "vB39_par_unite.csv")
    b = R[(R.jeu == "batteries") & R.methode.isin(METH)].copy()
    b["entr"] = b.cas.str.extract(r"entr\. (\S+) →")[0]
    b["test"] = b.cas.str.extract(r"→ (\S+)")[0]
    temp = lambda g: "40C" if "40C" in g else "TA"                                   # noqa: E731
    b["meme_T"] = [temp(a) == temp(c) for a, c in zip(b.entr, b.test)]
    c = b.groupby(["methode", "entr", "test", "meme_T"]).rho.median().reset_index()
    paires = c.pivot_table(index=["entr", "test"], columns="methode", values="rho")[METH].round(2)
    tout = c.groupby(["methode", "meme_T"]).rho.median().unstack().reindex(METH).round(2)
    ta = c[~c.entr.str.contains("40C")].groupby(["methode", "meme_T"]).rho.median().unstack().reindex(METH).round(2)
    # les paires à même température sont surtout uniforme ↔ biaisée : on croise avec le profil de courant
    prof = lambda g: "uni" if g == "uniforme" else ("basse" if "basse" in g else "haute")  # noqa: E731
    c["profil"] = ["uniforme<->biaisee" if "uni" in (prof(a), prof(x)) else
                   ("meme" if prof(a) == prof(x) else "basse<->haute") for a, x in zip(c.entr, c.test)]
    c["change_T"] = ~c.meme_T
    croise = c.pivot_table(index=["profil", "change_T"], columns="methode", values="rho",
                           aggfunc="median")[METH].round(2)
    txt = ("Par paire (médiane des unités)\n" + paires.to_string()
           + "\n\nMédiane des paires, même température ambiante (True) ou non\n" + tout.to_string()
           + "\n\nIdem, entraînement à température ambiante seulement\n" + ta.to_string()
           + "\n\nProfil de courant × changement de température (médiane des 4 paires par case)\n" + croise.to_string())
    (RES / "vB72.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
