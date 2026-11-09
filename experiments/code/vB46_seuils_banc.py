"""vB46 — Seuils d'alarme pour toutes les méthodes du banc, N-CMAPSS (R16 / R3 de PAPIER_ICDE § 8).

La relecture (W5) : « les seuils ne se transfèrent pour aucune méthode » n'était étayé que pour H-CDE et la SFA. Ici,
les 17 méthodes de vB39 (mêmes 12 cas N-CMAPSS, mêmes scores) passent par l'évaluateur commun INCHANGÉ : seuil =
quantile 0,95 des vols sains de l'unité de référence (classe vue, D-23) ; fausses alarmes = part des vols sains au-dessus
du seuil ; rappel = part des vols dégradés au-dessus du seuil ; par unité, puis médiane des unités non vues par cas,
graines agrégées comme vB39. Zone frontière (seuil T2) : lecture seule, annoncée au canal (M-33).
Lecture fixée AVANT de mesurer : le seuil « se transfère » pour une méthode si, en médiane sur les 12 cas, les fausses
alarmes sur les non vues sont ≤ 0,10 (le double du nominal) ET le rappel sur les non vues est ≥ rappel sur la
référence − 0,10. Pas d'attendu déclaré.

Machine : Mac. Sortie : results/vB46_seuils.csv, vB46.txt
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB39_banc_par_unite as b39  # noqa: E402


def tache(args):
    from evaluateur import evaluer
    import vB05_derive_usage as d5
    ds, ref = args
    v.it10.DILATION = v.parse_key(ds)[1]
    df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
    fc = df.groupby("unit").Fc.first()
    train = [u for u in sorted(fc[fc == 1].index) if u != ref]
    unseen = sorted(fc[fc != 1].index)
    complets = {u: v.fenetres(df, u) for u in [ref] + unseen}
    tr = pd.concat([d5.fenetres_derive(df, u, None)[0] for u in train], ignore_index=True)
    rows = []
    for nom, f in b39.tout(tr[v.S_COLS].values, tr[v.W_COLS].values, tr.unit.values, tr.cycle.values).items():
        m, g = b39._cle(nom)
        fen = pd.concat([x.assign(score=f(x[v.S_COLS].values, x[v.W_COLS].values))
                         for x in list(complets.values()) + [tr]], ignore_index=True)
        res = evaluer(fen, ds=ds, train_units=train, ref_unit=ref, non_vues=unseen)
        rows += [dict(cas=f"{ds} aucune ref u{ref}", methode=m, graine=g, unite=r.unite, role=r.role,
                      FA=r.FA_classe, rappel=r.rappel) for r in res.itertuples()]
    return rows


def main():
    import vB05_derive_usage as d5
    taches = []
    for ds in d5.DS_:
        fc = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv", usecols=["unit", "Fc"]).groupby("unit").Fc.first()
        taches += [(ds, ref) for ref in sorted(fc[fc == 1].index)]
    with Pool(int(os.environ.get("VB46_PROC", "10"))) as pool:
        L = pd.DataFrame([r for rr in pool.map(tache, taches, chunksize=1) for r in rr])
    L.to_csv(v.OUT / "vB46_seuils.csv", index=False)
    c = L.groupby(["cas", "methode", "graine", "role"])[["FA", "rappel"]].median().reset_index()
    c = c.groupby(["cas", "methode", "role"])[["FA", "rappel"]].median().reset_index()
    T = c.pivot_table(index=["cas", "methode"], columns="role", values=["FA", "rappel"]).groupby("methode").median()
    T.columns = [f"{a}_{b}" for a, b in T.columns]
    T["transfère"] = (T.FA_non_vue <= 0.10) & (T.rappel_non_vue >= T.rappel_ref - 0.10)
    with open(v.OUT / "vB46.txt", "w") as fh:
        for x in ["=== seuil de la référence (classe vue) appliqué aux non vues : médianes sur les 12 cas ===",
                  T.round(3).sort_values("FA_non_vue").to_string(),
                  f"\nméthodes dont le seuil se transfère : {list(T.index[T['transfère']]) or 'aucune'}"]:
            print(x)
            print(x, file=fh)


if __name__ == "__main__":
    main()
