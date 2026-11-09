"""vB02b — Trois questions de Martin sur vB02 (sondes supervisées = diagnostic, DB-01).

  (a) SUPPORT : à quel point les conditions de vol des unités non vues sont-elles loin de Fc1 ? Qu'a
      DS05 u9 de différent ? Par unité : moyennes de W, part des fenêtres hors étendue Fc1
      d'entraînement, et dépassement médian au-delà de cette étendue, en écarts-types Fc1.
  (b) AUTRES MODES DE PANNE : la sonde `lin` de vB02 (apprise sur les unités d'entraînement du DS,
      appliquée aux autres classes) sur DS04 (fan) et DS07 (LPT), qui n'ont pas de run H-CDE valide.
  (c) ENTRE MODES DE PANNE : une sonde apprise sur un DS (un mode) lit-elle la dégradation d'un AUTRE
      mode ? Spearman par vol, signé (une lecture inversée est une information).

Descriptif (D-15). Machine : Mac. Sortie : voie_B_avec_exo/results/vB02b_*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB02_sonde_transfert as t2  # noqa: E402
from evaluateur import _joindre, etiquettes, evaluer, par_vol  # noqa: E402

CLES = {"DS01": "DS01", "DS04": "DS04_d66", "DS05": "DS05", "DS07": "DS07"}   # fenêtres de la campagne v3
F = v.S_COLS + v.W_COLS


def charger(ds):
    v.it10.DILATION = v.parse_key(CLES[ds])[1]
    fc, train, ref, unseen = v.PLAN[ds]
    df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
    lab = df.groupby(["unit", "cycle"]).degradation.first().abs()
    f = {u: v.fenetres(df, u).drop(columns="cycle_debut") for u in list(train) + [ref] + list(unseen)}
    lignes_vol = df.groupby("unit").size() / df.groupby("unit").cycle.nunique()
    return dict(fc=fc, train=train, ref=ref, unseen=list(unseen), lab=lab, f=f, lignes_vol=lignes_vol)


def spearman_vol(ds, x, score):
    e = par_vol(_joindre(x[["unit", "cycle"]].assign(score=score), etiquettes(ds)))
    return spearmanr(e.score, e.degradation).statistic


def main():
    D = {ds: charger(ds) for ds in CLES}

    # (a) support
    rows = []
    for ds, d in D.items():
        tr = pd.concat([d["f"][u] for u in d["train"]])
        for u in [d["ref"]] + d["unseen"]:
            x = d["f"][u]
            r = dict(DS=ds, unite=u, role="ref" if u == d["ref"] else "non_vue",
                     classe=f"Fc{d['fc'] if u == d['ref'] else v.PLAN[ds][3][u]}",
                     lignes_par_vol=round(d["lignes_vol"][u]),
                     degr_max=round(float(d["lab"].loc[u].max()), 4))
            for c in v.W_COLS:
                lo, hi, sd = tr[c].min(), tr[c].max(), tr[c].std()
                exces = np.maximum(x[c] - hi, lo - x[c]).clip(0) / sd
                r[f"{c[2:]}_moy"] = round(x[c].mean(), 3)
                r[f"{c[2:]}_hors%"] = round(100 * (exces > 0).mean())
                r[f"{c[2:]}_exces_sd"] = round(float(np.median(exces)), 2)
            rows.append(r)
        rows.append(dict(DS=ds, unite="train", classe=f"Fc{d['fc']}",
                         lignes_par_vol=round(d["lignes_vol"][list(d["train"])].mean()),
                         degr_max=round(float(d["lab"].loc[list(d["train"])].max()), 4),
                         **{f"{c[2:]}_moy": round(tr[c].mean(), 3) for c in v.W_COLS}))
    sup = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("=== (a) support des conditions de vol ===")
    print(sup.to_string(index=False))
    sup.to_csv(v.OUT / "vB02b_support.csv", index=False)

    # (b) sonde lin de vB02 sur DS04 et DS07
    print("\n=== (b) transfert de classe, sonde lin, autres modes ===")
    res = []
    for ds in ["DS04", "DS07"]:
        d = D[ds]
        tr = pd.concat([d["f"][u] for u in d["train"]], ignore_index=True)
        s = t2.sondes()["lin"].fit(tr[F].values, d["lab"].loc[list(zip(tr.unit, tr.cycle))].values)
        fen = pd.concat([x.assign(score=s.predict(x[F].values)) for x in d["f"].values()])
        r = evaluer(fen, ds=ds, train_units=d["train"], ref_unit=d["ref"], non_vues=d["unseen"])
        res.append(r.assign(DS=ds))
    b = pd.concat(res)
    print(b[["DS", "unite", "role", "AUROC", "corr_spearman", "reponse_relative", "FA_classe", "rappel",
             "delai_frac", "non_detecte"]].round(3).to_string(index=False))
    b.to_csv(v.OUT / "vB02b_transfert_autres_modes.csv", index=False)

    # (c) entre modes : apprendre sur DS_a (unités d'entraînement), lire DS_b (réf. + non vues)
    print("\n=== (c) entre modes de panne : Spearman par vol, signé (ligne = appris sur, colonne = lu sur) ===")
    rows = []
    for a, da in D.items():
        tr = pd.concat([da["f"][u] for u in da["train"]], ignore_index=True)
        s = t2.sondes()["lin"].fit(tr[F].values, da["lab"].loc[list(zip(tr.unit, tr.cycle))].values)
        for b_, db in D.items():
            for u in [db["ref"]] + db["unseen"]:
                x = db["f"][u]
                rows.append(dict(appris=f"{a}", lu=f"{b_}", unite=u,
                                 rho=spearman_vol(b_, x, s.predict(x[F].values))))
    c = pd.DataFrame(rows)
    c.to_csv(v.OUT / "vB02b_entre_modes.csv", index=False)
    print(c.pivot_table(index="appris", columns=["lu", "unite"], values="rho").round(2).to_string())


if __name__ == "__main__":
    main()
