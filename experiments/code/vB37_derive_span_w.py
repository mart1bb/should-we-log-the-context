"""vB37 — Condition C1 (contexte plus rapide que la dégradation) cassée : le contexte aide-t-il ? vB05c refait avec les
méthodes du papier.

vB05c a montré que la lenteur est piégée par une dérive d'usage et qu'une SFA insensible aux conditions résiste à une
dérive modérée — mais avec `insens_W_v99` et une SFA réduite à 99 %, pas les méthodes du papier. Fig. 4 (vB36, à
partir de vB27) ne montre pas d'avantage de `span_W` sous dérive modérée. On refait le test proprement.

Méthodes (celles du papier, inchangées) : `lenteur` (`sfa.ajuster`), `span_W` v2 (DB-08), `resid_quad` (modèle de
contexte quadratique + SFA). Dérive d'usage fabriquée dans les unités d'entraînement (vB05) : `montante_h0.30`
(modérée, ~45 vols par unité), `montante_h0.15` et `descendante_h0.15` (fortes, ~25 vols), plus `aucune`.
**Contrôle de même taille, sans dérive** : pour chaque scénario dérivé, les mêmes nombres de vols par unité tirés au
hasard (graine 0) — sépare l'effet de la dérive de celui du manque de vols (vB05b).
Découpages : DS01, DS05, DS07 ; chaque unité Fc1 tour à tour en référence, les autres Fc1 en entraînement ; non vues
= toutes les Fc2/Fc3 (complètes). Mesure : Spearman par vol avec la dégradation (évaluateur commun), médiane sur les
non vues ; et R²(score ~ W) médian sur les non vues (le score suit-il l'usage ?).

Lecture fixée AVANT de mesurer, par scénario dérivé (médiane sur les 12 découpages) :
  - « la lenteur est piégée par la dérive » si lenteur(dérive) ≤ lenteur(contrôle même taille) − 0,10 ;
  - « le contexte aide quand C1 casse » si, en plus, span_W(dérive) ≥ lenteur(dérive) + 0,05 ;
  - sinon on décrit (et le papier dit que rien ne marche bien quand C1 casse).

Machine : Mac (Martin, 12 oct.). Sortie : voie_B_avec_exo/results/vB37_*.csv, vB37.txt
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB05_derive_usage as d5  # noqa: E402
import vB27_robustesse_decoupages as r27  # noqa: E402

DERIVES = ["montante_h0.30", "montante_h0.15", "descendante_h0.15"]
METH = ["lenteur", "resid_quad", "span_W"]


def tache(args):
    from evaluateur import evaluer
    ds, ref, sc = args
    v.it10.DILATION = v.parse_key(ds)[1]
    df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
    fc = df.groupby("unit").Fc.first()
    train = [u for u in sorted(fc[fc == 1].index) if u != ref]
    unseen = sorted(fc[fc != 1].index)
    complets = {u: v.fenetres(df, u) for u in [ref] + unseen}
    if sc.startswith("controle "):
        derive = d5.SCENARIOS[sc.split()[1]]
        rng = np.random.default_rng(0)
        parts = []
        for u in train:
            du = df[df.unit == u]
            n = len(d5.vols_gardes(du, derive))
            vols = np.sort(rng.choice(du.cycle.unique(), n, replace=False))
            parts.append(v.fenetres(du[du.cycle.isin(vols)], u))
        tr = pd.concat(parts, ignore_index=True)
    else:
        tr = pd.concat([d5.fenetres_derive(df, u, d5.SCENARIOS[sc])[0] for u in train], ignore_index=True)
    S, W, un, cy = tr[v.S_COLS].values, tr[v.W_COLS].values, tr.unit.values, tr.cycle.values
    fs = r27.methodes(S, W, un, cy)
    fs["resid_quad"] = fs.pop("modele")              # nom du papier
    row = dict(DS=ds, ref=ref, scenario=sc, vols_par_unite=float(tr.groupby("unit").cycle.nunique().mean()))
    for nom in METH:
        f = fs[nom]
        fen = pd.concat([x.assign(score=f(x[v.S_COLS].values, x[v.W_COLS].values))
                         for x in list(complets.values()) + [tr]], ignore_index=True)
        res = evaluer(fen, ds=ds, train_units=train, ref_unit=ref, non_vues=unseen)
        row[nom] = float(res[res.role == "non_vue"].corr_spearman.median())
        r2 = []
        for u in unseen:
            m = (fen.unit == u).values
            Wu = fen.loc[m, v.W_COLS].values
            r2.append(LinearRegression().fit(Wu, fen.score[m]).score(Wu, fen.score[m]))
        row[f"R2W_{nom}"] = float(np.median(r2))
    return row


def main():
    taches = []
    for ds in d5.DS_:
        fc = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv", usecols=["unit", "Fc"]).groupby("unit").Fc.first()
        for ref in sorted(fc[fc == 1].index):
            taches += [(ds, ref, sc) for sc in ["aucune"] + DERIVES + [f"controle {s}" for s in DERIVES]]
    with Pool(10) as pool:
        t = pd.DataFrame(pool.map(tache, taches, chunksize=1))
    t.to_csv(v.OUT / "vB37_derive_span_w.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["vols_par_unite"] + METH + [f"R2W_{m}" for m in METH]
    print("=== médianes sur les 12 découpages, par scénario ===")
    print(t.groupby("scenario")[cols].median().round(2).to_string())
    print("\n=== par DS (Spearman médian) ===")
    print(t.pivot_table(index="DS", columns="scenario", values=METH, aggfunc="median").round(2).T.to_string())
    print("\n=== lecture fixée ===")
    med = t.groupby("scenario")[METH].median()
    for s in DERIVES:
        piege = med.loc[s, "lenteur"] <= med.loc[f"controle {s}", "lenteur"] - 0.10
        aide = med.loc[s, "span_W"] >= med.loc[s, "lenteur"] + 0.05
        print(f"{s:20s} lenteur {med.loc[s, 'lenteur']:.2f} (contrôle {med.loc[f'controle {s}', 'lenteur']:.2f}) "
              f"→ piégée : {'OUI' if piege else 'non'} ; span_W {med.loc[s, 'span_W']:.2f} → le contexte aide : "
              f"{'OUI' if (piege and aide) else 'non'}")


if __name__ == "__main__":
    main()
