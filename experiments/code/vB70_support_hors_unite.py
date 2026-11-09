"""vB70 — Distance au support avec une référence hors unité (reviews IA du 20 oct., R2-W3 et R1-Q4).

Défaut de vB49 : le seuil y est le quantile 0,99 de la distance d'une fenêtre d'entraînement à son plus proche AUTRE
voisin d'entraînement, presque toujours une fenêtre voisine de la même unité ; les fenêtres de test viennent d'autres
unités, donc le seuil est trop bas et « dans le support » N-CMAPSS sortait à 51 %.
Ici : seuil = quantile q des distances d'une fenêtre d'entraînement à son plus proche voisin dans une AUTRE unité
d'entraînement (q ∈ {0,95 ; 0,99 ; 0,999}). Tout le reste comme vB49 (W standardisé sur l'entraînement, ≤ 20 000
fenêtres tirées, graine 0, mêmes tâches, mêmes fenêtres de test retenues). L'ancien seuil est recalculé pour contrôle.
Mesure descriptive, sans réglage.

Machine : Mac. Sortie : results/vB70.txt, vB70_distance.csv
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB39_banc_par_unite as b39  # noqa: E402
import vB47_dans_le_support as b47  # noqa: E402

RELEVES: list = []
QS = (0.95, 0.99, 0.999)


def distance(S, W, unit, cycle, labels=None):
    mu, sd = W.mean(0), W.std(0) + 1e-12
    Z = (W - mu) / sd
    rng = np.random.default_rng(0)
    i = rng.choice(len(Z), min(20_000, len(Z)), replace=False)
    Zr, ur = Z[i], np.asarray(unit)[i]
    nn = NearestNeighbors(n_neighbors=2).fit(Zr)
    ancien = np.quantile(nn.kneighbors(Zr)[0][:, 1], 0.99) + 1e-12
    d_ref = np.full(len(Zr), np.nan)
    for u in np.unique(ur):
        m = ur == u
        if m.all():
            continue
        d_ref[m] = NearestNeighbors(n_neighbors=1).fit(Zr[~m]).kneighbors(Zr[m])[0][:, 0]
    d_ref = d_ref[np.isfinite(d_ref)]
    seuils = {q: np.quantile(d_ref, q) + 1e-12 for q in QS}

    def f(S_, W_):
        d = nn.kneighbors((W_ - mu) / sd, n_neighbors=1)[0][:, 0]
        r = dict(ancien=float((d > ancien).mean()))
        r.update({f"q{q}": float((d > s).mean()) for q, s in seuils.items()})
        r["rapport"] = float(np.median(d / seuils[0.99]))
        RELEVES.append(r)
        return d
    return {"distance": f}


def lancer(arg):
    quel, t = arg
    RELEVES.clear()
    if quel == "hors":
        b39.tout = distance
        rows = b39.lancer(t)
    else:
        b47.tout = distance
        rows = b47.lancer(t)
    jeu = rows[0]["jeu"] if rows else t[0]
    cas = rows[0]["cas"] if rows else str(t[1])
    if t[0] == "N":       # comme vB49 : référence (classe vue) notée en premier, puis non vues, puis l'entraînement
        r = pd.DataFrame(RELEVES[:1] if quel == "dans" else RELEVES[1:-1])
    else:
        r = pd.DataFrame(RELEVES)
    out = dict(quel=quel, jeu=jeu, cas=cas)
    out.update({k: r[k].mean() for k in ["ancien"] + [f"q{q}" for q in QS]})
    out["rapport"] = r.rapport.median()
    return out


def main():
    import vB42_quoi_journaliser as b42
    T = [("hors", t) for t in b42.taches_base() if t[0] != "C"] + [("dans", t) for t in b47.taches()]
    with Pool(10) as pool:
        R = pd.DataFrame(pool.map(lancer, T, chunksize=1))
    R.to_csv(v.OUT / "vB70_distance.csv", index=False)
    cols = ["ancien"] + [f"q{q}" for q in QS] + ["rapport"]
    t = R.groupby(["jeu", "quel"])[cols].median().round(3)
    print(t.to_string())
    (v.OUT / "vB70.txt").write_text(t.to_string() + "\n")


if __name__ == "__main__":
    main()
