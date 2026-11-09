"""vB49 — Distance au support des contextes de test (2e relecture, Q6). Mesure descriptive.

Pour chaque cas (hors support : tâches de vB39 ; dans le support : tâches de vB47), W standardisé sur l'entraînement ;
d(x) = distance euclidienne d'une fenêtre au plus proche voisin parmi les fenêtres d'entraînement (≤ 20 000 tirées,
graine 0) ; seuil = quantile 0,99 des distances d'une fenêtre d'entraînement à son plus proche AUTRE voisin
d'entraînement. Rapporté : part des fenêtres de test au-delà du seuil (« hors support »), médiane des d / seuil.
Les fenêtres d'entraînement que les tâches N-CMAPSS notent aussi (en dernier) sont écartées.
C-MAPSS exclu : le régime retenu est absent de l'entraînement par construction, et ses tâches notent aussi les
autres régimes.

Machine : Mac. Sortie : results/vB49.txt, vB49_distance.csv
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


def distance(S, W, unit, cycle, labels=None):
    mu, sd = W.mean(0), W.std(0) + 1e-12
    Z = (W - mu) / sd
    rng = np.random.default_rng(0)
    Zr = Z[rng.choice(len(Z), min(20_000, len(Z)), replace=False)]
    nn = NearestNeighbors(n_neighbors=2).fit(Zr)
    seuil = np.quantile(nn.kneighbors(Zr)[0][:, 1], 0.99) + 1e-12

    def f(S_, W_):
        d = nn.kneighbors((W_ - mu) / sd, n_neighbors=1)[0][:, 0]
        RELEVES.append(dict(hors=float((d > seuil).mean()), rapport=float(np.median(d / seuil))))
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
    if t[0] == "N":       # N-CMAPSS : référence (classe vue) notée en premier, puis non vues, puis l'entraînement
        r = pd.DataFrame(RELEVES[:1] if quel == "dans" else RELEVES[1:-1])
    else:
        r = pd.DataFrame(RELEVES)
    return dict(quel=quel, jeu=jeu, cas=cas, part_hors=r.hors.mean(), rapport=r.rapport.median())


def main():
    import vB42_quoi_journaliser as b42
    T = [("hors", t) for t in b42.taches_base() if t[0] != "C"] + [("dans", t) for t in b47.taches()]   # C-MAPSS : régime retenu absent par construction
    with Pool(10) as pool:
        R = pd.DataFrame(pool.map(lancer, T, chunksize=1))
    R.loc[R.jeu == "C-MAPSS", "jeu"] = "C-MAPSS"
    R.to_csv(v.OUT / "vB49_distance.csv", index=False)
    t = R.groupby(["jeu", "quel"]).part_hors.median().unstack("quel").round(3)   # rapport : sans objet si W discret (gaz)
    print(t.to_string())
    (v.OUT / "vB49.txt").write_text(t.to_string() + "\n")


if __name__ == "__main__":
    main()
