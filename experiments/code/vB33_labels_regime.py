"""vB33 — Les labels de régime servent-ils, même à une méthode faite pour s'en servir ? (généralisation de domaine)

Question du papier B (5 oct.) : faut-il collecter les labels de régime ? Après vB32 (un a priori temporel sans
contexte égale ou dépasse le contexte sur 4 jeux informatifs sur 5), on ajoute deux méthodes qui CONSOMMENT un label
de régime discret, à la manière de la généralisation de domaine (non supervisée : aucune étiquette de santé) :
  - `norm_regime` : capteurs standardisés par régime (moyenne et écart-type du régime sur l'entraînement), puis
    lenteur — la pratique courante sur C-MAPSS (capteurs constants à l'intérieur d'un régime retirés :
    nuls après normalisation, ex. 6 capteurs de C-MAPSS) ; un vol de test prend les statistiques du régime d'entraînement le
    plus proche en W standardisé (régime jamais vu → statistiques empruntées) ;
  - `invariant_regime` : lenteur dans le complément de l'espace des écarts de moyenne entre régimes (capteurs
    standardisés ; directions couvrant 99 % de la dispersion inter-régimes, règle de `span_W`) — le score est
    aveugle aux décalages de régime vus : équivalent linéaire d'un apprentissage adversarial de domaine (DANN).
    Complément vide → méthode non définie (NaN). Le cas non linéaire n'est pas traité (limite écrite).
Labels de régime : C-MAPSS = régimes natifs ; ailleurs, k-moyennes (K = 6, graine 0) sur W standardisé de
l'entraînement — imitation d'un label de régime discret qu'un industriel collecterait.
Jeux, découpages, vérités : ceux de vB32 (mêmes intitulés de cas) ; la lenteur est recalculée ici (contrôle), les
tendances sont relues dans vB32_tendance_contre_lenteur.csv.

Lecture fixée AVANT de mesurer : par cas, gain = max(norm_regime, invariant_regime) − max(lenteur, tendance_moy,
tendance_reg) ; par jeu, médiane des gains. « Les labels servent » sur un jeu si la médiane ≥ 0,05. Attendu (déclaré) :
seulement sur les batteries (C3). PVDAQ rapporté à part (non informatif, vB31).

Machine : Mac (Martin, 5 oct.). Sortie : voie_B_avec_exo/results/vB33_*.csv, vB33.txt
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
from sklearn.cluster import KMeans  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB08_espace_conditions as v8  # noqa: E402
import sfa  # noqa: E402

K = 6
METH = ["lenteur", "norm_regime", "invariant_regime"]


def methodes(S, W, unit, cycle, labels=None):
    """Scoreurs (S, W) → score. `labels` : régimes natifs de l'entraînement, sinon k-moyennes sur W."""
    muW, sdW = W.mean(0), W.std(0) + 1e-12
    Wz = (W - muW) / sdW
    lab = np.asarray(labels) if labels is not None else KMeans(K, n_init=10, random_state=0).fit_predict(Wz)
    regs = np.unique(lab)
    cent = np.array([Wz[lab == r].mean(0) for r in regs])
    m_r = np.array([S[lab == r].mean(0) for r in regs])
    s_r = np.array([S[lab == r].std(0) for r in regs])
    m_r_tous = m_r.copy()

    def assigner(W_):
        Wz_ = (W_ - muW) / sdW
        return np.argmin(((Wz_[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)

    idx_tr = np.searchsorted(regs, lab)
    garde = (s_r > 1e-8).all(axis=0)              # capteur constant dans un régime : nul après normalisation
    m_r, s_r = m_r[:, garde], s_r[:, garde]
    m_norm = sfa.ajuster((S[:, garde] - m_r[idx_tr]) / s_r[idx_tr], unit, cycle)
    # invariant : complément des écarts de moyenne entre régimes (capteurs standardisés)
    mu, sd = S.mean(0), S.std(0) + 1e-12
    k = S.shape[1]
    Mz = (m_r_tous - mu) / sd
    poids = np.array([(lab == r).mean() for r in regs])
    B = (Mz * np.sqrt(poids)[:, None])
    val, vec = np.linalg.eigh(B.T @ B)
    val, vec = val[::-1], vec[:, ::-1]
    r = int(np.searchsorted(np.cumsum(val) / val.sum(), sfa.VAR_CIBLE) + 1)
    N = np.linalg.svd(np.eye(k) - vec[:, :r] @ vec[:, :r].T)[0][:, :k - r]
    fv = sfa.moyennes_par_vol((S - mu) / sd, unit, cycle)
    w = v8.sfa_complement(fv, k, N)
    m_len = sfa.ajuster(S, unit, cycle)
    return {"lenteur": lambda S_, W_: m_len.score(S_),
            "norm_regime": lambda S_, W_: m_norm.score((S_[:, garde] - m_r[assigner(W_)]) / s_r[assigner(W_)]),
            "invariant_regime": lambda S_, W_: ((S_ - mu) / sd) @ w}


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
    for nom, f in methodes(tr[v.S_COLS].values, tr[v.W_COLS].values, tr.unit.values, tr.cycle.values).items():
        fen = pd.concat([x.assign(score=f(x[v.S_COLS].values, x[v.W_COLS].values))
                         for x in list(complets.values()) + [tr]], ignore_index=True)
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
    Wk = [c for c in b19.W_C if tr[c].std() > 1e-8]
    fs = methodes(tr[X].values.astype(float), tr[Wk].values.astype(float), tr.unit.values, tr.cycle.values,
                  labels=tr.regime.values)
    out = {}
    for nom, f in fs.items():
        t = te.assign(s=f(te[X].values.astype(float), te[Wk].values.astype(float)))
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
    fs = methodes(tr[b22.S_B].values, tr[b22.W_B].values, tr.bat.values, tr.cycle.values)
    for nom, f in fs.items():
        t = te.assign(s=f(te[b22.S_B].values, te[b22.W_B].values))
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
        Wk = [c for c in u29.W_U if c == "T_amb" or tr[f"set_{c}"].nunique() > 1]
        fs = methodes(tr[u29.S_U].values, tr[Wk].values, tr.unit.values, tr.cycle.values)
        for g_te in u29.GROUPES:
            if g_te == g_tr:
                continue
            te = df[(df.groupe == g_te) & ~df.unit.isin(u29.ENTR_SEULEMENT)].reset_index(drop=True)
            out = {}
            for nom, f in fs.items():
                t_ = te.assign(s=f(te[u29.S_U].values, te[Wk].values))
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
            fs = methodes(tr[g30.S_G].values, tr[["logc"]].values, tr.unit.values, tr.cycle.values)
            for nom, f in fs.items():
                t_ = te.assign(s=f(te[g30.S_G].values, te[["logc"]].values))
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
            fs = methodes(tr[p31.S_P].values, tr[p31.W_P].values, tr.unit.values, tr.cycle.values)
            for nom, f in fs.items():
                j = te.assign(s=f(te[p31.S_P].values, te[p31.W_P].values)).groupby(["unit", "age"]).s.mean()
                j = j.reset_index()
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
    t32 = pd.read_csv(v.OUT / "vB32_tendance_contre_lenteur.csv")
    t = t.merge(t32[["jeu", "cas", "lenteur", "tendance_moy", "tendance_reg"]].rename(columns={"lenteur": "lenteur_32"}),
                on=["jeu", "cas"], how="left")
    t["sans_contexte"] = t[["lenteur", "tendance_moy", "tendance_reg"]].max(axis=1)
    t["avec_labels"] = t[["norm_regime", "invariant_regime"]].max(axis=1)
    t["gain"] = t.avec_labels - t.sans_contexte
    t.to_csv(v.OUT / "vB33_labels_regime.csv", index=False)

    pd.set_option("display.width", 250)
    ordre = ["N-CMAPSS", "C-MAPSS", "batteries", "roulements", "gaz", "PVDAQ"]
    cols = ["lenteur", "lenteur_32", "tendance_moy", "tendance_reg", "norm_regime", "invariant_regime", "gain"]
    g = t.groupby("jeu")[cols].median().reindex(ordre)
    g["cas"] = t.groupby("jeu").size().reindex(ordre)
    g["invariant_NaN"] = t.groupby("jeu").invariant_regime.apply(lambda x: int(x.isna().sum())).reindex(ordre)
    print("=== médianes par jeu ===")
    print(g.round(3).to_string())
    servent = g.drop(index="PVDAQ").gain >= 0.05
    print(f"\njeux (hors PVDAQ) où les labels de régime servent (gain médian ≥ 0,05) : "
          f"{list(servent[servent].index) or 'aucun'}")


if __name__ == "__main__":
    main()
