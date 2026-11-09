"""vB47 — Contrôles DANS LE SUPPORT pour tous les jeux (2e relecture, W1).

La relecture : « le retrait du contexte se dégrade hors support » n'est établi que si les mêmes méthodes marchent
quand le contexte de test est vu à l'entraînement. Seul C-MAPSS avait ces contrôles (`C0_*`). Ici, mêmes méthodes
que vB39 (`vB39.tout`, mêmes réglages et graines), mêmes mesures, avec des découpages où le test reste dans le
support des contextes d'entraînement, unités de test jamais vues :
  - N-CMAPSS : l'unité de référence de chaque cas de vB39 (classe Fc1, hors entraînement) → 12 cas ;
  - batteries : dans chaque groupe d'usage, chaque cellule tour à tour en test, entraînement sur les autres
    cellules du même groupe → 24 cas ;
  - roulements : dans chaque groupe d'usage, chaque roulement de test (hors B01, B05) tour à tour, entraînement sur
    les autres roulements du groupe → 15 cas ;
  - gaz : par gaz, entraînement sur les concentrations basses des capteurs impairs (pairs), test sur les
    concentrations basses des capteurs pairs (impairs) → 12 cas ;
  - PVDAQ : les 8 contrôles de vB31 (3 systèmes d'un site en entraînement, le 4e en test).
Lecture fixée avant de mesurer (usage interne) : pour chaque méthode et chaque jeu, baisse = médiane dans le
support − médiane hors support (vB39). « La baisse hors support est attribuable au changement de contexte » pour une
méthode de retrait sur un jeu si (i) dans le support elle est à 0,05 au plus de la lenteur dans le support, et (ii) sa
baisse dépasse celle de la lenteur de 0,10 ou plus. Sinon : « la méthode est déjà faible dans le support » (si (i)
échoue) ou « pas de baisse propre au support » (si (ii) échoue).

Machine : Mac. Sortie : results/vB47_par_unite.csv, vB47_cas.csv, vB47.txt
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
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB39_banc_par_unite as b39  # noqa: E402

lignes, tout = b39.lignes, b39.tout


def t_ncmapss(args):
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
    cas, out = f"{ds} aucune ref u{ref}", []
    for nom, f in tout(tr[v.S_COLS].values, tr[v.W_COLS].values, tr.unit.values, tr.cycle.values).items():
        fen = pd.concat([x.assign(score=f(x[v.S_COLS].values, x[v.W_COLS].values))
                         for x in list(complets.values()) + [tr]], ignore_index=True)
        res = evaluer(fen, ds=ds, train_units=train, ref_unit=ref, non_vues=unseen)
        r = res[res.role == "ref"]
        out += lignes("N-CMAPSS", cas, nom, dict(zip(r.unite, r.corr_spearman)))
    return out


def t_batteries(args):
    import vB22_batteries as b22
    g, test = args
    df = b22.charger()
    tr = df[df.bat.isin([b for b in b22.GROUPES[g] if b != test])]
    te = df[df.bat == test]
    out = []
    for nom, f in tout(tr[b22.S_B].values, tr[b22.W_B].values, tr.bat.values, tr.cycle.values).items():
        t = te.assign(s=f(te[b22.S_B].values, te[b22.W_B].values))
        fv = t.groupby("cycle").agg(s=("s", "mean"), cap=("capacite", "first"))
        out += lignes("batteries", f"{g} cellule {test}", nom, {test: spearmanr(fv.s, -fv.cap).statistic})
    return out


def t_roulements(args):
    import vB29_roulements_upb as u29
    g, test = args
    df = u29.charger()
    tr = df[df.unit.isin([u for u in u29.GROUPES[g] if u != test])]
    te = df[df.unit == test].reset_index(drop=True)
    Wk = [c for c in u29.W_U if c == "T_amb" or tr[f"set_{c}"].nunique() > 1]
    out = []
    for nom, f in tout(tr[u29.S_U].values, tr[Wk].values, tr.unit.values, tr.cycle.values).items():
        s = f(te[u29.S_U].values, te[Wk].values)
        out += lignes("roulements", f"{g} roulement {test}", nom, {test: spearmanr(te.cycle, s).statistic})
    return out


def t_gaz(args):
    import vB30_capteurs_gaz as g30
    gaz, impairs = args
    df = g30.charger()
    dg = df[df.gaz == gaz]
    bas = dg[dg.conc <= dg.conc.median()]
    u_tr = [u for u in range(1, 17) if (u % 2 == 1) == impairs]
    tr, te = bas[bas.unit.isin(u_tr)], bas[~bas.unit.isin(u_tr)].reset_index(drop=True)
    cas, out = f"gaz {gaz} {'impairs→pairs' if impairs else 'pairs→impairs'} (basses)", []
    for nom, f in tout(tr[g30.S_G].values, tr[["logc"]].values, tr.unit.values, tr.cycle.values).items():
        t_ = te.assign(s=f(te[g30.S_G].values, te[["logc"]].values))
        out += lignes("gaz", cas, nom, {un: spearmanr(g.lot, g.s).statistic for un, g in t_.groupby("unit")})
    return out


def t_pvdaq(args):
    import vB31_pvdaq as p31
    a, out_u = args
    df = p31.ch.charger()
    df = df[df.unit.isin(p31.SITES[a])].copy()
    df["age"] = (df.jour - df.groupby("unit").jour.transform("min")).dt.days
    tr, te = df[df.unit != out_u], df[df.unit == out_u].reset_index(drop=True)
    out = []
    for nom, f in tout(tr[p31.S_P].values, tr[p31.W_P].values, tr.unit.values, tr.cycle.values).items():
        j = te.assign(s=f(te[p31.S_P].values, te[p31.W_P].values)).groupby("age").s.mean()
        out += lignes("PVDAQ", f"contrôle {a} sans {out_u}", nom, {out_u: spearmanr(j.index, j.values).statistic})
    return out


def lancer(t):
    nom, args = t
    return {"N": t_ncmapss, "B": t_batteries, "R": t_roulements, "G": t_gaz, "P": t_pvdaq}[nom](args)


def taches():
    import vB05_derive_usage as d5
    import vB22_batteries as b22
    import vB29_roulements_upb as u29
    import vB31_pvdaq as p31
    t = []
    for ds in d5.DS_:
        fc = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv", usecols=["unit", "Fc"]).groupby("unit").Fc.first()
        t += [("N", (ds, ref)) for ref in sorted(fc[fc == 1].index)]
    t += [("B", (g, b)) for g, bs in b22.GROUPES.items() for b in bs]
    t += [("R", (g, u)) for g, us in u29.GROUPES.items() for u in us if u not in u29.ENTR_SEULEMENT]
    t += [("G", (gz, imp)) for gz in range(1, 7) for imp in (True, False)]
    t += [("P", (a, u)) for a in p31.SITES for u in p31.SITES[a]]
    return t


def main():
    T = taches()
    seul = os.environ.get("VB47_SEUL")
    if seul:
        T = [x for x in T if x[0] == seul][:1]
    T = sorted(T, key=lambda x: 0 if x[0] in ("N", "R", "P") else 1)
    with Pool(int(os.environ.get("VB47_PROC", "10"))) as pool:
        L = pd.DataFrame([r for rr in pool.map(lancer, T, chunksize=1) for r in rr])
    suf = f"_essai_{seul}" if seul else ""
    L.to_csv(v.OUT / f"vB47_par_unite{suf}.csv", index=False)
    C = b39.par_cas(L)
    C.to_csv(v.OUT / f"vB47_cas{suf}.csv", index=False)
    if seul:
        print(C.T.to_string())
        return
    hors = pd.read_csv(v.OUT / "vB39_cas.csv")
    cm = hors.jeu == "C-MAPSS"
    dans_c = hors[cm & hors.cas.str.contains("C0_")].assign(jeu="C-MAPSS")
    hors = hors[~cm | hors.cas.str.contains(r" R\d")]
    dans = pd.concat([C, dans_c], ignore_index=True)
    ordre = ["N-CMAPSS", "C-MAPSS", "batteries", "roulements", "gaz", "PVDAQ"]
    meths = [m for m in C.columns if m not in ("jeu", "cas")]
    md, mh = dans.groupby("jeu")[meths].median().reindex(ordre), hors.groupby("jeu")[meths].median().reindex(ordre)
    retrait = ["resid_lin", "resid_quad", "residu_FNN", "resid_GP", "resid_GP_z", "norm_regime"]
    with open(v.OUT / "vB47.txt", "w") as fh:
        def p(*a):
            print(*a)
            print(*a, file=fh)
        pd.set_option("display.width", 250)
        p("=== médiane DANS le support / HORS support (vB39) ===")
        p(pd.concat({"dans": md.T, "hors": mh.T}, axis=1).round(2).to_string())
        p("\n=== lecture fixée, méthodes de retrait ===")
        for j in ordre:
            if j == "PVDAQ":
                continue
            bl = md.loc[j, "lenteur"] - mh.loc[j, "lenteur"]
            for m in retrait:
                b = md.loc[j, m] - mh.loc[j, m]
                ok_i = md.loc[j, m] >= md.loc[j, "lenteur"] - 0.05
                ok_ii = b >= bl + 0.10
                v_ = ("attribuable au support" if ok_i and ok_ii else
                      "déjà faible dans le support" if not ok_i else "pas de baisse propre au support")
                p(f"  {j:11s} {m:12s} dans {md.loc[j, m]:+.2f} hors {mh.loc[j, m]:+.2f} baisse {b:+.2f} "
                  f"(lenteur {bl:+.2f}) → {v_}")


if __name__ == "__main__":
    main()
