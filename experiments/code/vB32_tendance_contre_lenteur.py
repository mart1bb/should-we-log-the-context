"""vB32 — Est-ce la lenteur, ou n'importe quel a priori temporel ? « tendance » contre lenteur sur les 7 jeux.

Critique (5 oct.) : sur plusieurs jeux la vérité est l'âge ; une méthode qui extrait une tendance monotone dans le
temps est avantagée, que ce soit la lenteur ou autre chose. On compare la lenteur à deux méthodes « tendance »,
sans contexte et sans étiquettes de santé (seul l'ordre des vols des unités d'entraînement est utilisé, comme la
SFA) :
  - `tendance_moy` : capteurs standardisés, centrés par unité ; pente de chaque capteur contre le rang du vol dans
    la vie de l'unité (0–1), sur les moyennes par vol ; direction = moyenne des pentes sur les unités ; score = S·a ;
  - `tendance_reg` : régression ridge (α = 1) du rang du vol sur les capteurs standardisés, centrés par unité
    (moyennes par vol) ; score = S·w. C'est « prédire le temps » : sur les jeux dont la vérité est l'âge, elle
    optimise sur l'entraînement la cible de l'évaluation (réserve déclarée).
Lenteur = `sfa.ajuster` (comme partout). Toutes orientées pour croître avec l'ordre sur l'entraînement.
Jeux et découpages : ceux des expériences précédentes, recalculés ici pour les trois méthodes —
  N-CMAPSS DS01/05/07 (vB27 : 24 découpages, vérité = dégradation) ; C-MAPSS FD002/FD004 (vB19 : 12 cas × 5 tirages,
  âge sur le régime cible) ; batteries (vB27 : 5 entraînements × 4 groupes, capacité) ; roulements (vB29 : 12 cas,
  âge) ; gaz (vB30 : 12 cas, lot) ; PVDAQ (vB31 : 8 cas inter-sites, âge).

Lecture fixée AVANT de mesurer, par jeu, sur la médiane par cas de (méthode tendance − lenteur) :
  - « la lenteur est spéciale » si, pour les deux tendances, l'écart médian ≤ −0,05 sur au moins 4 jeux sur 6
    (PVDAQ compté à part : non informatif en vB31) ;
  - « tout a priori temporel suffit » si l'écart médian est dans ± 0,05 sur au moins 4 jeux sur 6 pour au moins une
    tendance ;
  - sinon on décrit. On rapporte aussi les jeux où une tendance dépasse la lenteur de plus de 0,05.

Machine : Mac (Martin, 5 oct.). Sortie : voie_B_avec_exo/results/vB32_*.csv, vB32.txt
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
from sklearn.linear_model import Ridge  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import sfa  # noqa: E402

METH = ["lenteur", "tendance_moy", "tendance_reg"]


def methodes(S, unit, cycle):
    """Trois scoreurs S → score appris sur l'entraînement (S, unit, cycle)."""
    mu, sd = S.mean(0), S.std(0) + 1e-12
    k = S.shape[1]
    fv = sfa.moyennes_par_vol((S - mu) / sd, unit, cycle)
    X = fv[list(range(k))].values
    X = X - fv.groupby("unit")[list(range(k))].transform("mean").values
    rang = fv.groupby("unit").cycle.rank(pct=True).values
    rang_c = rang - pd.Series(rang).groupby(fv.unit.values).transform("mean").values
    pentes = [np.polyfit(rang[m], X[m], 1)[0] for m in [(fv.unit == u).values for u in fv.unit.unique()]
              if m.sum() >= 3]
    a = np.mean(pentes, axis=0)
    w = Ridge(alpha=1.0, fit_intercept=False).fit(X, rang_c).coef_
    m_len = sfa.ajuster(S, unit, cycle)
    return {"lenteur": lambda S_: m_len.score(S_),
            "tendance_moy": lambda S_: ((S_ - mu) / sd) @ a,
            "tendance_reg": lambda S_: ((S_ - mu) / sd) @ w}


# --------------------------------------------------------------------------------------------------
def t_ncmapss(args):
    import vB05_derive_usage as d5
    from evaluateur import evaluer
    ds, sc, ref = args
    v.it10.DILATION = v.parse_key(ds)[1]
    df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
    fc = df.groupby("unit").Fc.first()
    train = [u for u in sorted(fc[fc == 1].index) if u != ref]
    unseen = sorted(fc[fc != 1].index)
    complets = {u: v.fenetres(df, u) for u in [ref] + unseen}
    tr = pd.concat([d5.fenetres_derive(df, u, d5.SCENARIOS[sc])[0] for u in train], ignore_index=True)
    out = {}
    for nom, f in methodes(tr[v.S_COLS].values, tr.unit.values, tr.cycle.values).items():
        fen = pd.concat([x.assign(score=f(x[v.S_COLS].values)) for x in list(complets.values()) + [tr]],
                        ignore_index=True)
        res = evaluer(fen, ds=ds, train_units=train, ref_unit=ref, non_vues=unseen)
        out[nom] = float(res[res.role == "non_vue"].corr_spearman.median())
    return [dict(jeu="N-CMAPSS", cas=f"{ds} {sc} ref u{ref}", **out)]


def t_cmapss(args):
    import vB19_cmapss_avec_w as b19
    fd, cas, retire, cible, tirage = args
    d = b19.a6.charger(fd)
    rng = np.random.default_rng(tirage)
    u = d.unit.unique()
    tr_u = set(rng.choice(u, size=len(u) // 2, replace=False))
    tr = d[d.unit.isin(tr_u)]
    if retire is not None:
        tr = tr[tr.regime != retire]
    te = d[~d.unit.isin(tr_u)]
    X = [c for c in b19.a6.COLS[5:] if tr[c].std() > 1e-8]
    out = {}
    for nom, f in methodes(tr[X].values.astype(float), tr.unit.values, tr.cycle.values).items():
        t = te.assign(s=f(te[X].values.astype(float)))
        sp = [spearmanr(g[g.regime == cible].cycle, g[g.regime == cible].s).statistic
              for _, g in t.groupby("unit") if (g.regime == cible).sum() >= 10]
        out[nom] = float(np.median(sp))
    return [dict(jeu="C-MAPSS", cas=f"{fd} {cas} tirage {tirage}", **out)]


def t_batteries(g_tr):
    import vB22_batteries as b22
    df = b22.charger()
    df["groupe"] = df.bat.map({b: g for g, bs in b22.GROUPES.items() for b in bs})
    tr, te = df[df.groupe == g_tr], df[df.groupe != g_tr]
    sp = {}
    for nom, f in methodes(tr[b22.S_B].values, tr.bat.values, tr.cycle.values).items():
        t = te.assign(s=f(te[b22.S_B].values))
        fv = t.groupby(["groupe", "bat", "cycle"]).agg(s=("s", "mean"), cap=("capacite", "first")).reset_index()
        for (g, b), x in fv.groupby(["groupe", "bat"]):
            sp.setdefault(g, {}).setdefault(nom, []).append(spearmanr(x.s, -x.cap).statistic)
    return [dict(jeu="batteries", cas=f"entr. {g_tr} → {g}", **{n: float(np.median(x)) for n, x in d.items()})
            for g, d in sp.items()]


def t_roulements():
    import vB29_roulements_upb as u29
    df = u29.charger()
    rows = []
    for g_tr in u29.GROUPES:
        tr = df[df.groupe == g_tr]
        fs = methodes(tr[u29.S_U].values, tr.unit.values, tr.cycle.values)
        for g_te in u29.GROUPES:
            if g_te == g_tr:
                continue
            te = df[(df.groupe == g_te) & ~df.unit.isin(u29.ENTR_SEULEMENT)].reset_index(drop=True)
            out = {}
            for nom, f in fs.items():
                t_ = te.assign(s=f(te[u29.S_U].values))
                out[nom] = float(np.median([spearmanr(g.cycle, g.s).statistic for _, g in t_.groupby("unit")]))
            rows.append(dict(jeu="roulements", cas=f"{g_tr} → {g_te}", **out))
    return rows


def t_gaz():
    import vB30_capteurs_gaz as g30
    df = g30.charger()
    rows = []
    for gaz in sorted(df.gaz.unique()):
        dg = df[df.gaz == gaz]
        med = dg.conc.median()
        for nom_dec, impairs in [("impairs→pairs", True), ("pairs→impairs", False)]:
            u_tr = [u for u in range(1, 17) if (u % 2 == 1) == impairs]
            tr = dg[dg.unit.isin(u_tr) & (dg.conc <= med)]
            te = dg[~dg.unit.isin(u_tr) & (dg.conc > med)].reset_index(drop=True)
            out = {}
            for nom, f in methodes(tr[g30.S_G].values, tr.unit.values, tr.cycle.values).items():
                t_ = te.assign(s=f(te[g30.S_G].values))
                out[nom] = float(np.median([spearmanr(g.lot, g.s).statistic for _, g in t_.groupby("unit")]))
            rows.append(dict(jeu="gaz", cas=f"gaz {gaz} {nom_dec}", **out))
    return rows


def t_pvdaq():
    import vB31_pvdaq as p31
    df = p31.ch.charger()
    df = df[df.unit.isin(sum(p31.SITES.values(), []))].copy()
    df["age"] = (df.jour - df.groupby("unit").jour.transform("min")).dt.days
    rows = []
    for a, b in [("Las_Vegas", "Golden"), ("Golden", "Las_Vegas")]:
        for out_u in p31.SITES[a]:
            u_tr = [u for u in p31.SITES[a] if u != out_u]
            tr = df[df.unit.isin(u_tr)]
            te = df[df.unit.isin(p31.SITES[b])].reset_index(drop=True)
            out = {}
            for nom, f in methodes(tr[p31.S_P].values, tr.unit.values, tr.cycle.values).items():
                j = te.assign(s=f(te[p31.S_P].values)).groupby(["unit", "age"]).s.mean().reset_index()
                out[nom] = float(np.median([spearmanr(g.age, g.s).statistic for _, g in j.groupby("unit")]))
            rows.append(dict(jeu="PVDAQ", cas=f"{a}→{b} sans {out_u}", **out))
    return rows


def t_simple(nom):
    return {"roulements": t_roulements, "gaz": t_gaz, "PVDAQ": t_pvdaq}[nom]()


def main():
    import vB05_derive_usage as d5
    import vB19_cmapss_avec_w as b19
    import vB22_batteries as b22
    tn = []
    for ds in d5.DS_:
        fc = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv", usecols=["unit", "Fc"]).groupby("unit").Fc.first()
        tn += [(ds, sc, ref) for sc in ["aucune", "montante_h0.30"] for ref in sorted(fc[fc == 1].index)]
    tc = [(fd, cas, retire, cible, t) for fd in ["FD002", "FD004"] for cas, retire, cible in b19.a6.CAS
          for t in range(b19.N_TIRAGES)]
    with Pool(10) as pool:
        rn = pool.map_async(t_ncmapss, tn, chunksize=1)
        rc = pool.map_async(t_cmapss, tc, chunksize=1)
        rb = pool.map_async(t_batteries, list(b22.GROUPES), chunksize=1)
        rs = pool.map_async(t_simple, ["roulements", "gaz", "PVDAQ"], chunksize=1)
        rows = [r for rr in rn.get() + rc.get() + rb.get() + rs.get() for r in rr]
    t = pd.DataFrame(rows)
    for m in METH[1:]:
        t[f"ecart_{m}"] = t[m] - t.lenteur
    t.to_csv(v.OUT / "vB32_tendance_contre_lenteur.csv", index=False)

    pd.set_option("display.width", 250)
    ordre = ["N-CMAPSS", "C-MAPSS", "batteries", "roulements", "gaz", "PVDAQ"]
    g = t.groupby("jeu")[METH + ["ecart_tendance_moy", "ecart_tendance_reg"]].median().reindex(ordre)
    g["cas"] = t.groupby("jeu").size().reindex(ordre)
    print("=== médianes par jeu (performance contre la vérité du jeu) ===")
    print(g.round(3).to_string())
    info = g.drop(index="PVDAQ")
    E = ["ecart_tendance_moy", "ecart_tendance_reg"]
    speciale = ((info[E] <= -0.05).all(axis=1)).sum()
    suffit = max(int((info[e].abs() <= 0.05).sum()) for e in E)
    print(f"\njeux (hors PVDAQ) où la lenteur bat les deux tendances de ≥ 0,05 : {speciale} / {len(info)}")
    print(f"jeux où une tendance est à ± 0,05 de la lenteur (meilleure des deux tendances) : {suffit} / {len(info)}")
    print("lecture : " + ("la lenteur est spéciale" if speciale >= 4 else
                          "tout a priori temporel suffit" if suffit >= 4 else "à décrire"))
    print("\njeux où une tendance dépasse la lenteur de plus de 0,05 :")
    print(g[(g[E] > 0.05).any(axis=1)][METH + E].round(3).to_string() or "aucun")


if __name__ == "__main__":
    main()
