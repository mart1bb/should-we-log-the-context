"""vB39 — Banc du papier refait par unité, avec des méthodes de plus (révision après relecture, PAPIER_ICDE § 8).

Mêmes 124 cas et mêmes 12 méthodes que vB35 (fonction `vB35.toutes`, inchangée ; DANN réentraîné avec
`vB34.entrainer`, mêmes graines), plus :
  - R9 : la corrélation de Spearman est gardée PAR UNITÉ de test (et par graine pour les réseaux), pour les tests
    appariés et le bootstrap hiérarchique (vB40). Contrôle de cohérence : les médianes par cas doivent retrouver
    vB35 (écart ≤ 1e-6 ; réseaux : mêmes graines et même sous-échantillon).
  - R10, indicateurs SANS contexte et SANS a priori temporel (le « sain » supposé = 20 % premiers de la vie de chaque
    unité d'entraînement, comme le résidu FNN ; aucune étiquette) :
      `acp_T2`  : ACP sur les fenêtres saines standardisées, composantes jusqu'à 95 % de variance (au plus k − 1) ;
                  score = T² de Hotelling ;
      `acp_SPE` : erreur de reconstruction hors de ces composantes (statistique Q) ;
      `autoencodeur` : MLP k → 64 → 64 → q → 64 → 64 → k (q = nombre de composantes de l'ACP), entraîné sur les
                  fenêtres saines (réglages de vB26 : Adam 1e-3, lots de 512, 60 epochs, ≤ 50 000 lignes,
                  3 graines, médiane) ; score = norme de l'erreur de reconstruction.
  - R12, retrait du contexte à EXTRAPOLATION CONTRÔLÉE : processus gaussien W → S (S standardisé) sur les fenêtres
    saines, 1 000 points tirés (graine 0), noyau constant × RBF anisotrope + bruit blanc, hyperparamètres par
    vraisemblance marginale, `normalize_y` :
      `resid_GP`   : norme du résidu ‖S − m(W)‖ (moyenne seule) ;
      `resid_GP_z` : norme du résidu divisé par l'écart-type prédictif (bruit compris) : hors du support, l'incertitude
                  croît et le résidu y pèse moins, au lieu d'être pris pour de la dégradation.
Aucun réglage sur les cas de test ; aucun réglage après mesure.

Lecture fixée AVANT de mesurer (par cas : écart apparié ; par jeu : médiane des écarts ; jeux informatifs = tous
sauf PVDAQ ; C-MAPSS : régimes retenus seulement) :
  - R10 : « ignorer le contexte suffit, l'a priori temporel n'est pas nécessaire » si max(acp_T2, acp_SPE,
    autoencodeur) ≥ meilleure sans contexte de vB35 (max de lenteur, tendance_moy, tendance_reg) − 0,05 sur au moins
    3 des 5 jeux informatifs ; sinon « c'est l'a priori temporel qui fait marcher le sans-contexte ». Attendu
    (déclaré) : la seconde lecture (hors support, l'écart au sain est dominé par le contexte).
  - R12 : « un modèle de contexte à extrapolation contrôlée aide » sur un jeu si max(resid_GP, resid_GP_z) ≥
    meilleure sans contexte + 0,05. Attendu (déclaré) : seulement sur les batteries.

Machine : choisie par Martin. Sortie : voie_B_avec_exo/results/vB39_par_unite.csv (long : jeu, cas, methode,
graine, unite, rho), vB39_cas.csv (médiane par cas, graines agrégées comme vB35), vB39.txt
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
import vB35_table_unifiee as t35  # noqa: E402

SAIN, VAR_ACP, N_GP, LOT_PRED = 0.2, 0.95, 1000, 20_000
GRAINES = t35.GRAINES_NN
NOUVELLES = ["acp_T2", "acp_SPE", "autoencodeur", "resid_GP", "resid_GP_z"]


def _rang(unit, cycle):
    return pd.Series(np.asarray(cycle)).groupby(np.asarray(unit)).rank(pct=True).values


def nouvelles(S, W, unit, cycle):
    """R10 et R12 : (S_, W_) → score, appris sur l'entraînement (fenêtres « saines » = 20 % premiers de la vie)."""
    import torch
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    unit, cycle = np.asarray(unit), np.asarray(cycle)
    rang = _rang(unit, cycle)
    mu, sd = S.mean(0), S.std(0) + 1e-12
    Z = (S - mu) / sd
    sain = rang < SAIN
    out = {}
    # ACP sur le sain
    Zr = Z[sain]
    m0 = Zr.mean(0)
    _, sv, Vt = np.linalg.svd(Zr - m0, full_matrices=False)
    lam = sv ** 2 / (len(Zr) - 1)
    q = int(np.searchsorted(np.cumsum(lam) / lam.sum(), VAR_ACP) + 1)
    q = max(1, min(q, S.shape[1] - 1))
    P, lq = Vt[:q].T, lam[:q] + 1e-12
    out["acp_T2"] = lambda S_, W_: ((((S_ - mu) / sd - m0) @ P) ** 2 / lq).sum(1)

    def spe(S_, W_):
        X = (S_ - mu) / sd - m0
        return ((X - X @ P @ P.T) ** 2).sum(1)
    out["acp_SPE"] = spe
    # autoencodeur sur le sain (même sous-échantillon que les réseaux de vB35)
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(S), t35.MAX_NN, replace=False)) if len(S) > t35.MAX_NN else np.arange(len(S))
    Xs = torch.tensor(Z[sel][sain[sel]], dtype=torch.float32)
    k = S.shape[1]
    for g in GRAINES:
        import vB26_baselines as b26
        torch.set_num_threads(1)
        torch.manual_seed(g)
        rg = np.random.default_rng(g)
        nn = torch.nn
        ae = nn.Sequential(nn.Linear(k, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, q),
                           nn.Linear(q, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, k))
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
        for _ in range(b26.EPOCHS):
            for i in np.array_split(rg.permutation(len(Xs)), max(1, len(Xs) // b26.LOT)):
                opt.zero_grad()
                ((ae(Xs[i]) - Xs[i]) ** 2).mean().backward()
                opt.step()
        ae.eval()
        out[f"autoencodeur_g{g}"] = (lambda S_, W_, ae=ae: torch.linalg.norm(
            torch.tensor((S_ - mu) / sd, dtype=torch.float32) - ae(torch.tensor((S_ - mu) / sd, dtype=torch.float32)),
            dim=1).detach().numpy())
    # processus gaussien W → S sur le sain
    muW, sdW = W.mean(0), W.std(0) + 1e-12
    Wz = (W - muW) / sdW
    i_s = np.where(sain)[0]
    i_s = np.sort(np.random.default_rng(0).choice(i_s, min(N_GP, len(i_s)), replace=False))
    noyau = ConstantKernel(1.0) * RBF(length_scale=np.ones(W.shape[1])) + WhiteKernel(0.1)
    gp = GaussianProcessRegressor(noyau, normalize_y=True, random_state=0).fit(Wz[i_s], Z[i_s])

    def pred(W_):
        Wq = (W_ - muW) / sdW
        m, s = [], []
        for j in range(0, len(Wq), LOT_PRED):
            a, b = gp.predict(Wq[j:j + LOT_PRED], return_std=True)
            m.append(a.reshape(len(Wq[j:j + LOT_PRED]), -1))
            s.append(b.reshape(len(Wq[j:j + LOT_PRED]), -1))
        return np.vstack(m), np.vstack(s)

    def gp_brut(S_, W_):
        m, _ = pred(W_)
        return np.linalg.norm((S_ - mu) / sd - m, axis=1)

    def gp_z(S_, W_):
        m, s = pred(W_)
        return np.linalg.norm(((S_ - mu) / sd - m) / (s + 1e-12), axis=1)
    out["resid_GP"], out["resid_GP_z"] = gp_brut, gp_z
    return out


def dann(S, W, unit, cycle, labels=None):
    """DANN de vB34 (dann_lent), mêmes graines et mêmes labels ; seul le réseau avec adversaire est réentraîné."""
    import vB34_dann_regime as d34
    import vB33_labels_regime as m33
    from sklearn.cluster import KMeans
    unit, cycle = np.asarray(unit), np.asarray(cycle)
    mu, sd = S.mean(0), S.std(0) + 1e-12
    Sz = (S - mu) / sd
    Wz = (W - W.mean(0)) / (W.std(0) + 1e-12)
    lab = (pd.factorize(np.asarray(labels))[0] if labels is not None
           else KMeans(m33.K, n_init=10, random_state=0).fit_predict(Wz))
    ordre = np.lexsort((np.arange(len(S)), cycle, unit))
    u_o = unit[ordre]
    meme = u_o[1:] == u_o[:-1]
    idx_a, idx_b = ordre[:-1][meme], ordre[1:][meme]
    rang = _rang(unit, cycle)
    out = {}
    for g in d34.GRAINES:
        f = d34.entrainer(Sz, idx_a, idx_b, lab, True, g, rang)
        out[f"DANN_g{g}"] = (lambda S_, W_, f=f: f((S_ - mu) / sd))
    return out


def tout(S, W, unit, cycle, labels=None):
    f = t35.toutes(S, W, unit, cycle, labels=labels)
    f.update(dann(S, W, unit, cycle, labels=labels))
    f.update(nouvelles(S, W, unit, cycle))
    return f


def _cle(nom):
    """'residu_FNN_g1' → ('residu_FNN', 1) ; 'lenteur' → ('lenteur', -1)."""
    if "_g" in nom and nom.rsplit("_g", 1)[1].isdigit():
        m, g = nom.rsplit("_g", 1)
        return m, int(g)
    return nom, -1


def lignes(jeu, cas, nom, rhos):
    m, g = _cle(nom)
    return [dict(jeu=jeu, cas=cas, methode=m, graine=g, unite=str(u), rho=float(r)) for u, r in rhos.items()]


# --------------------------------------------------------------------------------------------------
# Tâches : copies de vB35 qui gardent la corrélation par unité (mêmes données, mêmes découpages).
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
        r = res[res.role == "non_vue"]
        out += lignes("N-CMAPSS", cas, nom, dict(zip(r.unite, r.corr_spearman)))
    return out


def t_cmapss(args):
    import vB19_cmapss_avec_w as b19
    fd, cas_, retire, cible, tirage = args
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
    cas, out = f"{fd} {cas_} tirage {tirage}", []
    fs = tout(tr[X].values.astype(float), tr[Wk].values.astype(float), tr.unit.values, tr.cycle.values,
              labels=tr.regime.values)
    for nom, f in fs.items():
        t = te.assign(s=f(te[X].values.astype(float), te[Wk].values.astype(float)))
        rh = {un: spearmanr(g[g.regime == cible].cycle, g[g.regime == cible].s).statistic
              for un, g in t.groupby("unit") if (g.regime == cible).sum() >= 10}
        out += lignes("C-MAPSS", cas, nom, rh)
    return out


def t_batteries(g_tr):
    import vB22_batteries as b22
    df = b22.charger()
    df["groupe"] = df.bat.map({b: g for g, bs in b22.GROUPES.items() for b in bs})
    tr, te = df[df.groupe == g_tr], df[df.groupe != g_tr]
    out = []
    for nom, f in tout(tr[b22.S_B].values, tr[b22.W_B].values, tr.bat.values, tr.cycle.values).items():
        t = te.assign(s=f(te[b22.S_B].values, te[b22.W_B].values))
        fv = t.groupby(["groupe", "bat", "cycle"]).agg(s=("s", "mean"), cap=("capacite", "first")).reset_index()
        for g, xg in fv.groupby("groupe"):
            rh = {b: spearmanr(x.s, -x.cap).statistic for b, x in xg.groupby("bat")}
            out += lignes("batteries", f"entr. {g_tr} → {g}", nom, rh)
    return out


def t_roulements(g_tr):
    import vB29_roulements_upb as u29
    df = u29.charger()
    tr = df[df.groupe == g_tr]
    Wk = [c for c in u29.W_U if c == "T_amb" or tr[f"set_{c}"].nunique() > 1]
    fs = tout(tr[u29.S_U].values, tr[Wk].values, tr.unit.values, tr.cycle.values)
    out = []
    for g_te in u29.GROUPES:
        if g_te == g_tr:
            continue
        te = df[(df.groupe == g_te) & ~df.unit.isin(u29.ENTR_SEULEMENT)].reset_index(drop=True)
        for nom, f in fs.items():
            t_ = te.assign(s=f(te[u29.S_U].values, te[Wk].values))
            out += lignes("roulements", f"{g_tr} → {g_te}", nom,
                          {un: spearmanr(g.cycle, g.s).statistic for un, g in t_.groupby("unit")})
    return out


def t_gaz(args):
    import vB30_capteurs_gaz as g30
    gaz, impairs = args
    df = g30.charger()
    dg = df[df.gaz == gaz]
    med = dg.conc.median()
    u_tr = [u for u in range(1, 17) if (u % 2 == 1) == impairs]
    tr = dg[dg.unit.isin(u_tr) & (dg.conc <= med)]
    te = dg[~dg.unit.isin(u_tr) & (dg.conc > med)].reset_index(drop=True)
    cas, out = f"gaz {gaz} {'impairs→pairs' if impairs else 'pairs→impairs'}", []
    for nom, f in tout(tr[g30.S_G].values, tr[["logc"]].values, tr.unit.values, tr.cycle.values).items():
        t_ = te.assign(s=f(te[g30.S_G].values, te[["logc"]].values))
        out += lignes("gaz", cas, nom, {un: spearmanr(g.lot, g.s).statistic for un, g in t_.groupby("unit")})
    return out


def t_pvdaq(args):
    import vB31_pvdaq as p31
    a, b, out_u = args
    df = p31.ch.charger()
    df = df[df.unit.isin(sum(p31.SITES.values(), []))].copy()
    df["age"] = (df.jour - df.groupby("unit").jour.transform("min")).dt.days
    tr = df[df.unit.isin([u for u in p31.SITES[a] if u != out_u])]
    te = df[df.unit.isin(p31.SITES[b])].reset_index(drop=True)
    cas, out = f"{a}→{b} sans {out_u}", []
    for nom, f in tout(tr[p31.S_P].values, tr[p31.W_P].values, tr.unit.values, tr.cycle.values).items():
        j = te.assign(s=f(te[p31.S_P].values, te[p31.W_P].values)).groupby(["unit", "age"]).s.mean().reset_index()
        out += lignes("PVDAQ", cas, nom, {un: spearmanr(g.age, g.s).statistic for un, g in j.groupby("unit")})
    return out


def lancer(t):
    nom, args = t
    return {"N": t_ncmapss, "C": t_cmapss, "B": t_batteries, "R": t_roulements, "G": t_gaz, "P": t_pvdaq}[nom](args)


def par_cas(L):
    """Médiane par cas ; réseaux : médiane sur les graines des médianes par cas (comme vB35)."""
    c = L.groupby(["jeu", "cas", "methode", "graine"]).rho.median().reset_index()
    return c.groupby(["jeu", "cas", "methode"]).rho.median().unstack("methode").reset_index()


def main():
    import vB05_derive_usage as d5
    import vB19_cmapss_avec_w as b19
    import vB22_batteries as b22
    import vB29_roulements_upb as u29
    import vB31_pvdaq as p31
    taches = []
    for ds in d5.DS_:
        fc = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv", usecols=["unit", "Fc"]).groupby("unit").Fc.first()
        taches += [("N", (ds, ref)) for ref in sorted(fc[fc == 1].index)]
    taches += [("C", (fd, cas, ret, cib, t)) for fd in ["FD002", "FD004"] for cas, ret, cib in b19.a6.CAS
               for t in range(b19.N_TIRAGES)]
    taches += [("B", g) for g in b22.GROUPES] + [("R", g) for g in u29.GROUPES]
    taches += [("G", (gz, imp)) for gz in range(1, 7) for imp in (True, False)]
    taches += [("P", (a, b, u)) for a, b in [("Las_Vegas", "Golden"), ("Golden", "Las_Vegas")] for u in p31.SITES[a]]
    seul = os.environ.get("VB39_SEUL")                      # essai rapide : une seule tâche, ex. « G »
    if seul:
        taches = [t for t in taches if t[0] == seul][:1]
    taches = [t for t in taches if t[0] in ("P", "R", "N")] + [t for t in taches if t[0] not in ("P", "R", "N")]
    with Pool(int(os.environ.get("VB39_PROC", "10"))) as pool:
        L = pd.DataFrame([r for rr in pool.map(lancer, taches, chunksize=1) for r in rr])
    suffixe = f"_essai_{seul}" if seul else ""
    L.to_csv(v.OUT / f"vB39_par_unite{suffixe}.csv", index=False)
    C = par_cas(L)
    C.to_csv(v.OUT / f"vB39_cas{suffixe}.csv", index=False)

    pd.set_option("display.width", 300)
    print("=== contrôle de cohérence : médianes par cas contre vB35 ===")
    t = pd.read_csv(v.OUT / "vB35_table_unifiee.csv")
    anc = sum(t35.FAMILLES.values(), [])
    m = C.merge(t[["jeu", "cas"] + anc], on=["jeu", "cas"], suffixes=("", "_35"))
    for k in anc:
        print(f"  {k:18s} écart max {float((m[k] - m[k + '_35']).abs().max()):.2e}")
    if seul:
        print(C.round(3).T.to_string())
        return
    cm = C.jeu == "C-MAPSS"
    C.loc[cm & C.cas.str.contains(r" R\d"), "jeu"] = "C-MAPSS retenu"
    C.loc[cm & C.cas.str.contains("C0_"), "jeu"] = "C-MAPSS contrôle"
    ordre = ["N-CMAPSS", "C-MAPSS retenu", "C-MAPSS contrôle", "batteries", "roulements", "gaz", "PVDAQ"]
    print("\n=== médiane par jeu ===")
    print(C.groupby("jeu")[anc + NOUVELLES].median().reindex(ordre).T.round(2).to_string())
    cf = C[["lenteur", "tendance_moy", "tendance_reg"]].max(axis=1)
    C["R10"] = C[["acp_T2", "acp_SPE", "autoencodeur"]].max(axis=1) - cf
    C["R12"] = C[["resid_GP", "resid_GP_z"]].max(axis=1) - cf
    g = C.groupby("jeu")[["R10", "R12"]].median().reindex(ordre)
    print("\n=== lectures fixées (médiane des écarts appariés à la meilleure sans contexte) ===")
    print(g.round(3).to_string())
    inf = ["N-CMAPSS", "C-MAPSS retenu", "batteries", "roulements", "gaz"]
    n10 = int((g.loc[inf, "R10"] >= -0.05).sum())
    print(f"R10 : jeux informatifs où ACP/AE ≥ meilleure sans contexte − 0,05 : {n10}/5 → "
          + ("ignorer le contexte suffit" if n10 >= 3 else "c'est l'a priori temporel qui compte"))
    aide = [j for j in inf if g.loc[j, "R12"] >= 0.05]
    print(f"R12 : jeux où le GP aide (≥ +0,05) : {aide or 'aucun'}")


if __name__ == "__main__":
    main()
