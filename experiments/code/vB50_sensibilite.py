"""vB50 — Sensibilité de la lenteur et de span_W à leurs seuils (2e relecture, Q3). Mesure descriptive.

Mêmes cas hors support et même mesure que vB39 (tâches de vB39). Réglages :
  - lenteur : seuil de blanchiment `eps` de `sfa.direction_lente` ∈ {1e-8, 1e-6 (défaut), 1e-4} ;
  - span_W : part de variance des effets du contexte qui définit V, `sfa.VAR_CIBLE` ∈ {0,95 ; 0,99 (défaut) ; 0,999}.
Rapporté : médiane par jeu pour chaque réglage, et écart maximal au réglage par défaut. Un réglage qui ne laisse
aucune direction (erreur d'algèbre linéaire) est non défini pour ce cas et compté à part.

Machine : Mac. Sortie : results/vB50.txt, vB50_par_unite.csv
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB39_banc_par_unite as b39  # noqa: E402

EPS, VAR = (1e-8, 1e-6, 1e-4), (0.95, 0.99, 0.999)


def reglages(S, W, unit, cycle, labels=None):
    import sfa
    import vB27_robustesse_decoupages as r27
    out = {}
    defaut = sfa.direction_lente.__defaults__
    for e in EPS:
        sfa.direction_lente.__defaults__ = (e,) + defaut[1:]
        try:
            m = sfa.ajuster(S, unit, cycle)
            out[f"lenteur_eps{e:.0e}"] = lambda S_, W_, m=m: m.score(S_)
        except np.linalg.LinAlgError:                      # aucune direction ne passe le seuil : non défini
            pass
    sfa.direction_lente.__defaults__ = defaut
    v0 = sfa.VAR_CIBLE
    for q in VAR:
        sfa.VAR_CIBLE = q
        try:
            m = r27.v8.span_W(S, W, unit, cycle, reduire=False)[0]
            out[f"span_W_var{q}"] = lambda S_, W_, m=m: m.score(S_)
        except np.linalg.LinAlgError:
            pass
    sfa.VAR_CIBLE = v0
    return out


def lancer(t):
    b39.tout = reglages
    return b39.lancer(t)


def main():
    import vB42_quoi_journaliser as b42
    T = sorted(b42.taches_base(), key=lambda x: 0 if x[0] in ("N", "R", "P") else 1)
    with Pool(10) as pool:
        L = pd.DataFrame([r for rr in pool.map(lancer, T, chunksize=1) for r in rr])
    L.to_csv(v.OUT / "vB50_par_unite.csv", index=False)
    C = L.groupby(["jeu", "cas", "methode"]).rho.median().unstack("methode")
    med = C.groupby("jeu").median()
    lent = [c for c in med if c.startswith("lenteur")]
    span = [c for c in med if c.startswith("span_W")]
    med["écart max lenteur"] = med[lent].sub(med["lenteur_eps1e-06"], axis=0).abs().max(axis=1)
    med["écart max span_W"] = med[span].sub(med["span_W_var0.99"], axis=0).abs().max(axis=1)
    n = C.notna().groupby("jeu").sum().T
    txt = med.round(3).T.to_string() + "\n\ncas définis par réglage :\n" + n.to_string()
    print(txt)
    (v.OUT / "vB50.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
