"""vB62 — Le journal comme attribut de contexte (A), comparé à une détection automatique des dates (B), avec placebo.

Pourquoi (19 oct.) : les corrections à date de vB51/vB60 comparaient des médianes sur quelques cycles et confondaient
saut et contexte rapide (placebo vB61). Ici l'effet de chaque événement journalisé est estimé par régression sur
toute la vie de l'unité, en même temps qu'une tendance lisse de l'âge, éventuellement après retrait du contexte.

Estimation, par unité, capteur par capteur (lignes de l'unité) :
    r_j(c) = a + b·âge + d·âge² + Σ_e δ_e · g_e(c) + bruit,    âge = rang du cycle dans la vie (0 à 1)
  r = S brut (`est_brut`) ou r = S − f(W) (`est_ctx`, f quadratique en W standardisé, ajustée sur l'entraînement
  comme resid_quad). Puis S_j ← S_j − Σ_e δ_e · g_e : seule la discontinuité est retirée, la tendance reste.
  Pour la forme « marche × W », la base contient aussi W standardisé (effet principal) : correction de spécification
  après un essai sur un cas C-MAPSS et un cas roulements, avant tout lancement complet (sans effet principal,
  l'interaction absorbait tout l'effet du contexte après la date : placebo −1,0 à −1,3).
  Forme g_e selon ce que dit le journal :
    - date seule : marche 1[c ≥ t] sur tous les capteurs ;
    - date + catégorie : mesure → marche, tous capteurs ; production → marche × W standardisé, tous capteurs ;
      maintenance → aucun retrait, vies d'entraînement coupées (vB60) ;
    - date + catégorie + capteurs : même forme sur les seuls capteurs nommés (maintenance : marche sur les capteurs
      nommés et vies coupées).
B, détection : mêmes réponses « date seule », mais les dates viennent des données : par unité, sur les moyennes par
  cycle de r standardisées, statistique Σ_j t_j² du coefficient de marche du modèle ci-dessus à chaque cycle candidat
  (au plus 200 candidats régulièrement espacés, hors k premiers et derniers cycles), on garde les 2 maxima séparés
  d'au moins k cycles (le nombre d'événements, 2, est donné au détecteur).
Cas, catégories, injections, tirages, mesure : vB60 (mêmes graines). Placebo : données propres, journal tiré comme
  dans vB60, mêmes réponses (B : le détecteur trouve toujours 2 dates).
Lecture (usage interne, fixée avant de mesurer) :
  (1) placebo : une réponse « abîme » si effet médian (réponse − propre) ≤ −0,05 ;
  (2) injecté : gain = réponse − sans_journal ; valeur du journal = (réponse avec dates du journal) − (même réponse
      avec dates détectées) ; valeur de la catégorie, des capteurs : comme vB60 ; « compte » si ≥ +0,05 ;
  (3) contexte pour estimer : est_ctx − est_brut.

Machine : Mac. Sortie : results/vB62_cas.csv, vB62.txt
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
from sklearn.linear_model import LinearRegression  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB51_evenements_injectes as e1  # noqa: E402
import vB60_trois_categories as b60  # noqa: E402

NIVEAUX = ["date", "cat", "cat_capteurs"]
ESTIMATIONS = ["est_brut", "est_ctx"]
N_CAND = 200


class Contexte:
    def __init__(self, tr, S, W):
        import vB24_regle_choix as r24
        self.mu, self.sd = tr[W].values.mean(0), tr[W].values.std(0) + 1e-12
        self.phi = r24.phi
        self.reg = LinearRegression().fit(self.phi(self.z(tr[W].values)), tr[S].values)

    def z(self, Wv):
        return (Wv - self.mu) / self.sd

    def residu(self, df, S, W):
        return df[S].values - self.reg.predict(self.phi(self.z(df[W].values)))


def forme(cat, niveau):
    """(type de g, capteurs : 'tous' ou 'nommes') ; None = aucun retrait."""
    if niveau == "date":
        return ("marche", "tous")
    if cat == "maintenance":
        return None if niveau == "cat" else ("marche", "nommes")
    g = "marcheW" if cat == "production" else "marche"
    return (g, "tous" if niveau == "cat" else "nommes")


def retirer(df, S, W, ctx, ev, f, est):
    """Estime et retire la discontinuité de chaque événement (voir en-tête)."""
    if f is None:
        return df
    g_type, quels = f
    df = df.copy()
    X = df[S].values.astype(float)
    R = ctx.residu(df, S, W) if est == "est_ctx" else X
    Z = ctx.z(df[W].values)
    unit, cyc = df.unit.values, df.cycle.values
    for u, liste in ev.items():
        m = np.where(unit == u)[0]
        if len(m) < 10 or not liste:
            continue
        c = cyc[m]
        age = pd.Series(c).rank(pct=True).values
        base = np.column_stack([np.ones(len(m)), age, age ** 2])
        if g_type == "marcheW":                             # interaction : l'effet principal du contexte est dans la base
            base = np.column_stack([base, Z[m]])
        blocs = []                                          # (colonnes g, capteurs concernés)
        for e in liste:
            pas = (c >= e["t"]).astype(float)
            G = pas[:, None] if g_type == "marche" else pas[:, None] * Z[m]
            blocs.append((G, set(range(len(S))) if quels == "tous" else set(e["capt"])))
        groupes = {}
        for j in range(len(S)):
            cle = tuple(i for i, (_, cs) in enumerate(blocs) if j in cs)
            if cle:
                groupes.setdefault(cle, []).append(j)
        for cle, js in groupes.items():
            G = np.column_stack([blocs[i][0] for i in cle])
            A = np.column_stack([base, G])
            coef = np.linalg.lstsq(A, R[np.ix_(m, js)], rcond=None)[0]
            X[np.ix_(m, js)] -= G @ coef[base.shape[1]:]
    df[S] = X
    return df


def detecter(df, S, W, ctx, est, n=2):
    """Dates détectées par unité (voir en-tête) ; renvoie un journal au format de vB60 (capteurs inconnus)."""
    R = ctx.residu(df, S, W) if est == "est_ctx" else df[S].values.astype(float)
    fv = pd.DataFrame(R).assign(unit=df.unit.values, cycle=df.cycle.values).groupby(["unit", "cycle"]).mean()
    ev = {}
    for u, g in fv.groupby(level=0):
        c = g.index.get_level_values(1).values
        Y = g.values
        Y = (Y - np.median(Y, 0)) / (np.std(np.diff(Y, axis=0), 0) + 1e-12)
        k = e1.fenetre(c)
        if len(c) < 3 * k:
            ev[u] = []
            continue
        age = np.linspace(0, 1, len(c))
        base = np.column_stack([np.ones(len(c)), age, age ** 2])
        cand = np.unique(np.linspace(k, len(c) - k - 1, min(N_CAND, len(c) - 2 * k)).astype(int))
        stat = []
        for p in cand:
            A = np.column_stack([base, (np.arange(len(c)) >= p).astype(float)])
            coef, res, *_ = np.linalg.lstsq(A, Y, rcond=None)
            s2 = (np.sum((Y - A @ coef) ** 2, 0) / max(len(c) - 4, 1)) + 1e-12
            var = s2 * np.linalg.pinv(A.T @ A)[-1, -1]
            stat.append(np.sum(coef[-1] ** 2 / var))
        stat = np.array(stat)
        choisis = []
        for i in np.argsort(stat)[::-1]:
            if all(abs(cand[i] - q) >= k for q in choisis):
                choisis.append(cand[i])
            if len(choisis) == n:
                break
        ev[u] = [dict(t=int(c[q]), capt=[], sg=[], r=0.0, a=[]) for q in sorted(choisis)]
    return ev


def appliquer(tr, te, S, W, cat, ev_tr, ev_te, niveau, est):
    """Réponse au niveau donné ; renvoie (entraînement, test) corrigés."""
    ctx = Contexte(tr, S, W)
    f = forme(cat, niveau)
    tr2, te2 = retirer(tr, S, W, ctx, ev_tr, f, est), retirer(te, S, W, ctx, ev_te, f, est)
    if cat == "maintenance" and niveau in ("cat", "cat_capteurs"):
        tr2 = b60.couper(tr2, ev_tr)
    return tr2, te2


def tache(args):
    jeu, cas, fonction, params = args
    tr, te, S, W = {"N": e1.cas_ncmapss, "C": e1.cas_cmapss, "B": e1.cas_batteries, "R": e1.cas_roulements}[fonction](*params)
    sig = e1.sigma_bruit(tr, S)
    mu, sd = tr[W].values.mean(0), tr[W].values.std(0) + 1e-12
    rows = [dict(jeu=jeu, cas=cas, mode="-", categorie="-", tirage=-1, reponse="propre", estimation="-",
                 **e1.mesurer(tr, te, S, W))]
    for mode in ("placebo", "injecte"):
        for cat in b60.CATEGORIES:
            for tg in b60.TIRAGES:
                rng = np.random.default_rng(1000 * tg + b60.CATEGORIES.index(cat))
                ev_tr, ev_te = b60.tirer(tr, S, W, cat, rng), b60.tirer(te, S, W, cat, rng)
                if mode == "injecte":
                    tr_i = b60.injecter(tr, S, W, cat, ev_tr, sig, mu, sd)
                    te_i = b60.injecter(te, S, W, cat, ev_te, sig, mu, sd)
                else:
                    tr_i, te_i = tr, te
                base = dict(jeu=jeu, cas=cas, mode=mode, categorie=cat, tirage=tg)
                if mode == "injecte":
                    rows.append(dict(base, reponse="sans_journal", estimation="-", **e1.mesurer(tr_i, te_i, S, W)))
                for est in ESTIMATIONS:
                    for niveau in NIVEAUX:
                        a, b = appliquer(tr_i, te_i, S, W, cat, ev_tr, ev_te, niveau, est)
                        rows.append(dict(base, reponse=niveau, estimation=est, **e1.mesurer(a, b, S, W)))
                    ctx = Contexte(tr_i, S, W)
                    d_tr, d_te = detecter(tr_i, S, W, ctx, est), detecter(te_i, S, W, ctx, est)
                    a, b = appliquer(tr_i, te_i, S, W, cat, d_tr, d_te, "date", est)
                    rows.append(dict(base, reponse="detection", estimation=est, **e1.mesurer(a, b, S, W)))
    if e1.ERREURS:
        print(f"{jeu} {cas} : {len(e1.ERREURS)} échecs cumulés dans ce processus", flush=True)
    return rows


def lire(R):
    meths = ["lenteur", "tendance_reg", "resid_quad", "span_W"]
    L = R.melt(id_vars=["jeu", "cas", "mode", "categorie", "tirage", "reponse", "estimation"], value_vars=meths,
               var_name="methode", value_name="rho")
    propre = L[L.reponse == "propre"].set_index(["jeu", "cas", "methode"]).rho
    L = L[L.reponse != "propre"].copy()
    L["propre"] = propre.reindex(pd.MultiIndex.from_frame(L[["jeu", "cas", "methode"]])).values
    sans = L[L.reponse == "sans_journal"].set_index(["jeu", "cas", "categorie", "tirage", "methode"]).rho
    L["sans"] = sans.reindex(pd.MultiIndex.from_frame(L[["jeu", "cas", "categorie", "tirage", "methode"]])).values
    P = L[L["mode"] == "placebo"].assign(effet=lambda x: x.rho - x.propre)
    I = L[(L["mode"] == "injecte") & (L.reponse != "sans_journal")].assign(gain=lambda x: x.rho - x.sans)
    tp = P.pivot_table(index=["jeu", "categorie", "methode"], columns=["estimation", "reponse"], values="effet",
                       aggfunc="median")
    ti = I.pivot_table(index=["jeu", "categorie", "methode"], columns=["estimation", "reponse"], values="gain",
                       aggfunc="median")
    dg = L[L.reponse == "sans_journal"].assign(d=lambda x: x.rho - x.propre).pivot_table(
        index=["jeu", "categorie", "methode"], values="d", aggfunc="median")
    return tp, ti, dg


def main():
    T = e1.taches()
    seul = os.environ.get("VB62_SEUL")
    if seul:
        T = [x for x in T if x[2] == seul][:1]
    T = sorted(T, key=lambda x: 0 if x[2] in ("N", "R") else 1)
    with Pool(int(os.environ.get("VB62_PROC", "10"))) as pool:
        R = pd.DataFrame([r for rr in pool.map(tache, T, chunksize=1) for r in rr])
    suf = f"_essai_{seul}" if seul else ""
    R.to_csv(v.OUT / f"vB62_cas{suf}.csv", index=False)
    tp, ti, dg = lire(R)
    pd.set_option("display.width", 300)
    txt = ("PLACEBO : effet médian (réponse − propre)\n" + tp.round(3).to_string()
           + "\n\nINJECTÉ : dégât sans journal (sans − propre)\n" + dg.round(3).to_string()
           + "\n\nINJECTÉ : gain médian (réponse − sans_journal)\n" + ti.round(3).to_string())
    print(txt)
    (v.OUT / f"vB62{suf}.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
