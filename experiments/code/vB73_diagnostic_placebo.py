"""vB73 — Diagnostic du placebo « maintenance, capteurs nommés + vies coupées » (vB62, vB71 : −0,4 à −0,6 sur N-CMAPSS).

Constat : sur données propres N-CMAPSS, la lenteur et span_W perdent ≈ 1,0 avec cette réponse, alors que couper seul
(0,00), la marche sur tous les capteurs (−0,06) et la marche sur un capteur nommé (−0,05) ne coûtent presque rien.
Hypothèse : la SFA calcule les différences entre vols consécutifs à l'intérieur de chaque unité (`sfa.direction_lente`).
Après coupe, le saut entre deux segments n'est plus pénalisé ; la marche estimée puis retirée sur données propres
laisse un faux saut à la date sur les capteurs nommés, qui devient une direction parfaitement lente.
Prédictions si l'hypothèse est juste : (a) avec coupe + marche nommée, la direction lente se concentre sur les
capteurs nommés (réfutée au premier passage : les capteurs nommés changent d'une unité à l'autre ; ajouté ensuite :
(a') le score d'entraînement saute aux dates de coupe, saut médian rapporté à l'écart-type des différences entre vols
consécutifs à l'intérieur des segments) ; (b) la perte disparaît si la SFA est ajustée sur les vies non coupées (mêmes données corrigées) ;
(c) avec la marche retirée sur tous les capteurs, la coupe ne coûte rien.
Cas : les 12 cas N-CMAPSS de vB60, placebo de la catégorie maintenance, tirages 0 et 1, estimation est_brut.

Sortie : results/vB73.txt
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import sfa  # noqa: E402
import vB01_sonde_marge as v  # noqa: E402
import vB51_evenements_injectes as e1  # noqa: E402
import vB60_trois_categories as b60  # noqa: E402
import vB62_journal_contexte as b62  # noqa: E402


def rho_lenteur(m, te, S):
    tev = te[te["eval"]]
    s = tev.assign(s=m.score(tev[S].values)).groupby(["unit", "cycle"]).agg(s=("s", "mean"), v=("verite", "first"))
    return float(np.median([spearmanr(g.s, g.v).statistic for _, g in s.groupby(level=0) if len(g) > 2]))


def saut(m, d, S, ev):
    """Saut médian du score d'entraînement aux dates du journal / écart-type des différences entre vols consécutifs."""
    fv = d.assign(s=m.score(d[S].values)).groupby(["unit", "cycle"]).s.mean()
    sauts, diffs = [], []
    for u, g in fv.groupby(level=0):
        c, x = g.index.get_level_values(1).values, g.values
        bornes = [e["t"] for e in ev.get(u, [])]
        seg = np.searchsorted(bornes, c, side="right")
        for i in range(seg.max() + 1):
            diffs.append(np.diff(x[seg == i]))
        for b in bornes:
            av, ap = x[c < b][-5:], x[c >= b][:5]
            if len(av) and len(ap):
                sauts.append(abs(ap.mean() - av.mean()))
    return float(np.median(sauts) / (np.std(np.concatenate(diffs)) + 1e-12))


def tache(args):
    jeu, cas, fonction, params = args
    tr, te, S, W = e1.cas_ncmapss(*params)
    out = []
    for tg in b60.TIRAGES:
        rng = np.random.default_rng(1000 * tg + b60.CATEGORIES.index("maintenance"))
        ev_tr, ev_te = b60.tirer(tr, S, W, "maintenance", rng), b60.tirer(te, S, W, "maintenance", rng)
        ctx = b62.Contexte(tr, S, W)
        conds = {
            "propre": (tr, False),
            "coupe seule": (tr, True),
            "marche nommée": (b62.retirer(tr, S, W, ctx, ev_tr, ("marche", "nommes"), "est_brut"), False),
            "marche nommée + coupe": (b62.retirer(tr, S, W, ctx, ev_tr, ("marche", "nommes"), "est_brut"), True),
            "marche tous + coupe": (b62.retirer(tr, S, W, ctx, ev_tr, ("marche", "tous"), "est_brut"), True),
        }
        te_c = {"marche nommée": b62.retirer(te, S, W, ctx, ev_te, ("marche", "nommes"), "est_brut"),
                "marche tous": b62.retirer(te, S, W, ctx, ev_te, ("marche", "tous"), "est_brut")}
        nommes = sorted({j for liste in ev_tr.values() for e in liste for j in e["capt"]})
        for nom, (d, coupe) in conds.items():
            d2 = b60.couper(d, ev_tr) if coupe else d
            m = sfa.ajuster(d2[S].values, d2.unit.values, d2.cycle.values)
            t = te_c["marche tous"] if "tous" in nom else (te_c["marche nommée"] if "nommée" in nom else te)
            # poids de la direction lente (entrées standardisées) sur les capteurs nommés de chaque unité
            part = []
            for u, liste in ev_tr.items():
                js = sorted({j for e in liste for j in e["capt"]})
                part.append(np.sum(m.w[js] ** 2))
            out.append(dict(cas=cas, tirage=tg, condition=nom, rho=rho_lenteur(m, t, S), saut=saut(m, d, S, ev_tr),
                            part_w_nommes_med=float(np.median(part)), n_capteurs=len(S), n_nommes_tous=len(nommes)))
        # (b) mêmes données corrigées (marche nommée), SFA ajustée sur les vies non coupées
        d = conds["marche nommée + coupe"][0]
        m = sfa.ajuster(d[S].values, d.unit.values, d.cycle.values)
        out.append(dict(cas=cas, tirage=tg, condition="marche nommée, SFA sur vies entières",
                        rho=rho_lenteur(m, te_c["marche nommée"], S), part_w_nommes_med=np.nan, n_capteurs=len(S),
                        n_nommes_tous=len(nommes)))
    return out


def main():
    T = [t for t in e1.taches() if t[2] == "N"]
    with Pool(10) as pool:
        R = pd.DataFrame([r for rr in pool.map(tache, T, chunksize=1) for r in rr])
    R.to_csv(v.OUT / "vB73.csv", index=False)
    t = R.groupby("condition")[["rho", "saut", "part_w_nommes_med"]].median().round(3)
    txt = ("N-CMAPSS, placebo maintenance, lenteur (médiane sur 12 cas × 2 tirages)\n" + t.to_string()
           + f"\n\npoids attendu sur 4 capteurs au hasard parmi {R.n_capteurs.iloc[0]} : "
           + f"{4 / R.n_capteurs.iloc[0]:.2f} (direction de norme 1)\n")
    (v.OUT / "vB73.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
