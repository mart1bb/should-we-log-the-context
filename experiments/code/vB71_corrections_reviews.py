"""vB71 — Expériences demandées par les reviews IA du 20 oct. (journal d'événements), à l'ampleur f = 1.

Cas, tirages, graines, unité d'injection (A_j), méthodes et mesure : ceux de vB67. Chaque expérience peut infirmer une
conclusion de la partie 2 du papier.

(1) Correction causale des unités de test (R2-W2 : l'équation (1) utilise toute la vie de l'unité de test). Les unités
    d'entraînement gardent l'estimation sur toute la vie (rétrospective, légitime). Sur l'unité de test, chaque
    événement e est estimé sur les cycles depuis l'événement précédent (ou le début) jusqu'à t_e + Δ, avec une
    tendance linéaire en cycle (au lieu de la quadratique en âge relatif, inconnu en production) ; Δ = k ou 3k,
    k = max(10, 5 % des cycles de l'unité). Formes, capteurs et estimations comme vB62. Réponses : date, cat,
    cat_capteurs. Placebo (données propres, mêmes dates) pour date et cat_capteurs.
(2) Capteurs retrouvés dans les données (R1-W4 : faut-il que le journal nomme les capteurs ?). Le journal donne date et
    catégorie ; à chaque date, modèle de vB62 sur tous les capteurs ; effet d'un capteur = moyenne quadratique de la
    correction estimée après la date / écart-type résiduel du capteur ; capteur retenu si son effet dépasse le
    quantile 0,95 de ses effets à 20 dates fictives de la même unité (tirées entre 10 et 90 % de la vie, à plus de k
    cycles des dates du journal, graine 0 ; chaque date fictive ajoutée seule au modèle). Puis correction « capteurs
    nommés » (aucune si aucun capteur n'est retenu). Réponse `cat_infere`, entraînement et test. Placebo aussi.
    Changement après l'essai sur un cas N-CMAPSS, avant tout lancement complet et sans regarder les gains : la
    première règle (meilleure coupure en deux du log de l'effet) retenait 10 capteurs sur 14 pour un événement à un
    capteur, et autant sur le placebo.
(3) Changement d'opération de forme différente de la correction (R1-W5, R2-W2). La correction reste marche × W̃
    (linéaire en W̃). Injection, sur 2 capteurs, f·A_j × h(W̃·a), h centrée réduite pour W̃·a ~ N(0, 1) :
      - `prod_quad` : h(x) = (x² − 1)/√2 ;
      - `prod_seuil` : h(x) = (max(x, 0) − 0,399)/0,584 (effet seulement d'un côté).
    Réponses : sans, date, cat, cat_capteurs, cat_infere, détection.
(4) Maintenance (R2-Q4, R1-Q3) : en plus des réponses de vB67, `capt_sans_couper` (marche sur les capteurs nommés,
    vies non coupées) et `date_couper` (marche sur tous les capteurs, vies coupées). Toutes les réponses sont aussi
    évaluées par segment : Spearman par segment entre deux dates du journal (`eval = segment`), ce qui mesure si le
    suivi reprend dans la nouvelle vie ; les scores ne dépendent pas de l'unité à l'inférence, donc c'est aussi
    « traiter chaque segment comme une nouvelle vie à l'inférence ».

Lecture (fixée avant de mesurer, comme vB68) : valeur d'un cas = médiane sur les 4 méthodes de (moyenne des 2 tirages
de réponse − sans_journal), meilleure des deux estimations par réponse ; par jeu : médiane sur les cas, IC bootstrap
95 %, Wilcoxon, Holm sur toutes les cases d'une même expérience. « Compte » : médiane ≥ +0,05 et Holm < 0,05.
Placebo : réponse − propre, « abîme » si ≤ −0,05.

Machine : Mac. Sorties : results/vB71_cas.csv, vB71_valeurs.csv, vB71.txt
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB51_evenements_injectes as e1  # noqa: E402
import vB60_trois_categories as b60  # noqa: E402
import vB62_journal_contexte as b62  # noqa: E402
import vB67_ampleurs_calibrees as b67  # noqa: E402

F = 1.0
METHODES = b67.METHODES
DELTAS = (1, 3)
VARIANTES = ["prod_quad", "prod_seuil"]


# ---------------------------------------------------------------------------------------------- injections
def produire_forme(df, S, W, ev, unite, mu, sd, h):
    df = df.copy()
    X = df[S].values.astype(float)
    Z = (df[W].values - mu) / sd
    for u, liste in ev.items():
        m_u = df.unit.values == u
        for e in liste:
            m = m_u & (df.cycle.values >= e["t"])
            for j, a in zip(e["capt"], e["a"]):
                X[m, j] += 3 * unite[j] * h(Z[m] @ a)
    df[S] = X
    return df


H = {"prod_quad": lambda x: (x ** 2 - 1) / np.sqrt(2),
     "prod_seuil": lambda x: (np.maximum(x, 0) - 0.399) / 0.584}


def injecter(df, S, W, cat, ev, unite, mu, sd):
    if cat in H:
        return produire_forme(df, S, W, ev, unite, mu, sd, H[cat])
    return b67.injecter(df, S, W, cat, ev, unite, mu, sd)


def cat_log(cat):
    """Catégorie inscrite au journal : les variantes sont des changements d'opération."""
    return "production" if cat in H else cat


# ---------------------------------------------------------------------------------------------- (1) causale
def retirer_causal(df, S, W, ctx, ev, f, est, mult):
    if f is None:
        return df
    g_type, quels = f
    df = df.copy()
    X = df[S].values.astype(float)
    R = ctx.residu(df, S, W) if est == "est_ctx" else X.copy()
    Z = ctx.z(df[W].values)
    unit, cyc = df.unit.values, df.cycle.values
    for u, liste in ev.items():
        m = np.where(unit == u)[0]
        if len(m) < 10 or not liste:
            continue
        c = cyc[m]
        cy = np.unique(c)
        k = e1.fenetre(cy)
        liste = sorted(liste, key=lambda e: e["t"])
        for i, e in enumerate(liste):
            t = e["t"]
            debut = liste[i - 1]["t"] if i > 0 else cy[0]
            fin = t + mult * k
            if i + 1 < len(liste):
                fin = min(fin, liste[i + 1]["t"])
            w = (c >= debut) & (c < fin)
            if (w & (c < t)).sum() < 3 or (w & (c >= t)).sum() < 3:
                continue
            pas = (c >= t).astype(float)
            G = pas[:, None] if g_type == "marche" else pas[:, None] * Z[m]
            js = list(range(len(S))) if quels == "tous" else list(e["capt"])
            base = np.column_stack([np.ones(len(m)), (c - t) / k])
            if g_type == "marcheW":
                base = np.column_stack([base, Z[m]])
            A = np.column_stack([base, G])
            coef = np.linalg.lstsq(A[w], R[np.ix_(m[w], js)], rcond=None)[0]
            corr = G @ coef[base.shape[1]:]
            X[np.ix_(m, js)] -= corr
            R[np.ix_(m, js)] -= corr
    df[S] = X
    return df


def appliquer_causal(tr, te, S, W, cat, ev_tr, ev_te, niveau, est, mult):
    ctx = b62.Contexte(tr, S, W)
    f = b62.forme(cat, niveau)
    tr2 = b62.retirer(tr, S, W, ctx, ev_tr, f, est)
    te2 = retirer_causal(te, S, W, ctx, ev_te, f, est, mult)
    if cat == "maintenance" and niveau in ("cat", "cat_capteurs"):
        tr2 = b60.couper(tr2, ev_tr)
    return tr2, te2


# ---------------------------------------------------------------------------------------------- (2) capteurs inférés
N_FICTIVES = 20


def inferer_capteurs(df, S, W, ctx, ev, g_type, est):
    R = ctx.residu(df, S, W) if est == "est_ctx" else df[S].values.astype(float)
    Z = ctx.z(df[W].values)
    unit, cyc = df.unit.values, df.cycle.values
    out = {}
    for u, liste in ev.items():
        m = np.where(unit == u)[0]
        if len(m) < 10 or not liste:
            out[u] = liste
            continue
        c = cyc[m]
        cy = np.unique(c)
        k = e1.fenetre(cy)
        age = pd.Series(c).rank(pct=True).values
        base = np.column_stack([np.ones(len(m)), age, age ** 2])
        if g_type == "marcheW":
            base = np.column_stack([base, Z[m]])
        Y = R[m]

        def g(t):
            pas = (c >= t).astype(float)
            return pas[:, None] if g_type == "marche" else pas[:, None] * Z[m]

        def effets(A, G, t):
            """Effet de G (dernières colonnes de A) par capteur, rapporté à l'écart-type résiduel."""
            coef = np.linalg.lstsq(A, Y, rcond=None)[0]
            sd_res = np.std(Y - A @ coef, axis=0) + 1e-12
            corr = G @ coef[-G.shape[1]:]
            return np.sqrt(np.mean(corr[c >= t] ** 2, axis=0)) / sd_res

        Gs = [g(e["t"]) for e in liste]
        A0 = np.column_stack([base] + Gs)
        lo, hi = cy[int(0.1 * len(cy))], cy[min(int(0.9 * len(cy)), len(cy) - 1)]
        cand = [t for t in cy if lo <= t <= hi and all(abs(t - e["t"]) > k for e in liste)]
        rng = np.random.default_rng(0)
        fict = rng.choice(cand, min(N_FICTIVES, len(cand)), replace=False) if len(cand) >= 5 else []
        nul = np.array([effets(np.column_stack([A0, g(t)]), g(t), t) for t in fict])
        seuil = np.quantile(nul, 0.95, axis=0) if len(nul) else np.full(len(S), np.inf)
        nouv = []
        for i, e in enumerate(liste):
            autres = [G for j, G in enumerate(Gs) if j != i]
            A = np.column_stack([base] + autres + [Gs[i]])
            eff = effets(A, Gs[i], e["t"])
            nouv.append(dict(e, capt=[int(j) for j in np.where(eff > seuil)[0]]))
        out[u] = nouv
    return out


def appliquer_infere(tr, te, S, W, cat, ev_tr, ev_te, est):
    ctx = b62.Contexte(tr, S, W)
    f = b62.forme(cat, "cat_capteurs")
    g_type = f[0]
    i_tr = inferer_capteurs(tr, S, W, ctx, ev_tr, g_type, est)
    i_te = inferer_capteurs(te, S, W, ctx, ev_te, g_type, est)
    tr2, te2 = b62.retirer(tr, S, W, ctx, i_tr, f, est), b62.retirer(te, S, W, ctx, i_te, f, est)
    if cat == "maintenance":
        tr2 = b60.couper(tr2, ev_tr)
    rappel = np.mean([len(set(a["capt"]) & set(b["capt"])) / max(len(b["capt"]), 1)
                      for u in ev_te for a, b in zip(i_te[u], ev_te[u])])
    nb = np.mean([len(a["capt"]) for u in i_te for a in i_te[u]])
    return tr2, te2, rappel, nb


# ---------------------------------------------------------------------------------------------- (4) maintenance
def appliquer_maint(tr, te, S, W, ev_tr, ev_te, variante, est):
    ctx = b62.Contexte(tr, S, W)
    if variante == "capt_sans_couper":
        return b62.retirer(tr, S, W, ctx, ev_tr, ("marche", "nommes"), est), \
            b62.retirer(te, S, W, ctx, ev_te, ("marche", "nommes"), est)
    tr2 = b60.couper(b62.retirer(tr, S, W, ctx, ev_tr, ("marche", "tous"), est), ev_tr)
    return tr2, b62.retirer(te, S, W, ctx, ev_te, ("marche", "tous"), est)


# ---------------------------------------------------------------------------------------------- tâche
def tache(args):
    jeu, cas, fonction, params = args
    tr, te, S, W = {"N": e1.cas_ncmapss, "C": e1.cas_cmapss, "B": e1.cas_batteries, "R": e1.cas_roulements}[fonction](*params)
    A = b67.amplitude(tr, S, W)
    mu, sd = tr[W].values.mean(0), tr[W].values.std(0) + 1e-12
    unite = F * A / 3
    rows = [dict(jeu=jeu, cas=cas, mode="-", categorie="-", tirage=-1, reponse="propre", estimation="-",
                 eval="global", **e1.mesurer(tr, te, S, W))]
    info = []
    cats = list(b60.CATEGORIES) + VARIANTES
    for cat in cats:
        cl = cat_log(cat)
        for tg in b60.TIRAGES:
            graine = 1000 * tg + (b60.CATEGORIES.index(cat) if cat in b60.CATEGORIES else 10 + VARIANTES.index(cat))
            rng = np.random.default_rng(graine)
            ev_tr, ev_te = b60.tirer(tr, S, W, cl, rng), b60.tirer(te, S, W, cl, rng)
            modes = ("injecte", "placebo") if cat in b60.CATEGORIES else ("injecte",)
            for mode in modes:
                if mode == "injecte":
                    tr_i, te_i = injecter(tr, S, W, cat, ev_tr, unite, mu, sd), injecter(te, S, W, cat, ev_te, unite, mu, sd)
                else:
                    tr_i, te_i = tr, te
                base = dict(jeu=jeu, cas=cas, mode=mode, categorie=cat, tirage=tg)
                segs = cl == "maintenance"

                def noter(rep, est, a, b):
                    rows.append(dict(base, reponse=rep, estimation=est, eval="global", **e1.mesurer(a, b, S, W)))
                    if segs:
                        rows.append(dict(base, reponse=rep, estimation=est, eval="segment",
                                         **e1.mesurer(a, b60.couper(b, ev_te), S, W)))

                if segs:                                    # référence propre évaluée par segment
                    rows.append(dict(base, reponse="propre", estimation="-", eval="segment",
                                     **e1.mesurer(tr, b60.couper(te, ev_te), S, W)))
                if mode == "injecte":
                    noter("sans_journal", "-", tr_i, te_i)
                for est in b62.ESTIMATIONS:
                    niveaux = b62.NIVEAUX if mode == "injecte" else ["date", "cat_capteurs"]
                    for niveau in niveaux:
                        if mode == "injecte":
                            a, b = b62.appliquer(tr_i, te_i, S, W, cl, ev_tr, ev_te, niveau, est)
                            noter(niveau, est, a, b)
                        for mult in DELTAS:
                            a, b = appliquer_causal(tr_i, te_i, S, W, cl, ev_tr, ev_te, niveau, est, mult)
                            noter(f"{niveau}_causal{mult}", est, a, b)
                    a, b, rap, nb = appliquer_infere(tr_i, te_i, S, W, cl, ev_tr, ev_te, est)
                    noter("cat_infere", est, a, b)
                    info.append(dict(base, estimation=est, rappel=rap, nb_capteurs=nb, n_capteurs=len(S)))
                    if segs and mode == "injecte":
                        for var in ("capt_sans_couper", "date_couper"):
                            a, b = appliquer_maint(tr_i, te_i, S, W, ev_tr, ev_te, var, est)
                            noter(var, est, a, b)
                if mode == "injecte" and cat in H:
                    ctx = b62.Contexte(tr_i, S, W)
                    d_tr, d_te = b62.detecter(tr_i, S, W, ctx, "est_brut"), b62.detecter(te_i, S, W, ctx, "est_brut")
                    a, b = b62.appliquer(tr_i, te_i, S, W, cl, d_tr, d_te, "date", "est_brut")
                    noter("detection", "est_brut", a, b)
    if e1.ERREURS:
        print(f"{jeu} {cas} : {len(e1.ERREURS)} échecs cumulés dans ce processus", flush=True)
    return rows, info


# ---------------------------------------------------------------------------------------------- lecture
def lire(R):
    L = R.melt(id_vars=["jeu", "cas", "mode", "categorie", "tirage", "reponse", "estimation", "eval"],
               value_vars=METHODES, var_name="methode", value_name="rho")
    P = L[L["mode"] != "-"].pivot_table(index=["jeu", "cas", "mode", "categorie", "eval", "methode"],
                                       columns=["reponse", "estimation"], values="rho", aggfunc="mean")  # 2 tirages
    propre_g = L[L["mode"] == "-"].set_index(["jeu", "cas", "methode"]).rho
    lignes = []
    for (mode, cat, ev), Pm in P.groupby(level=["mode", "categorie", "eval"]):
        Pm = Pm.dropna(axis=1, how="all")
        if ev == "segment":
            propre = Pm[("propre", "-")]
        else:
            propre = pd.Series(propre_g.reindex(Pm.index.droplevel(["mode", "categorie", "eval"])).values,
                               index=Pm.index)
        ref = Pm[("sans_journal", "-")] if mode == "injecte" else propre
        cols = {"degat": Pm[("sans_journal", "-")] - propre} if mode == "injecte" else {}
        for r in sorted({c[0] for c in Pm.columns} - {"sans_journal", "propre"}):
            cols[r] = Pm[[c for c in Pm.columns if c[0] == r]].max(axis=1) - ref   # meilleure des 2 estimations
        V = pd.DataFrame(cols).groupby(level=["jeu", "cas"]).median()               # médiane sur les méthodes
        for col in V.columns:
            for j, g in V[col].groupby(level="jeu"):
                lignes.append(dict(mode=mode, categorie=cat, eval=ev, jeu=j, valeur=col, **b67.resume(g.values)))
    T = pd.DataFrame(lignes)
    T["p_holm"] = np.nan
    for _, g in T.groupby(["mode", "eval"]):
        T.loc[g.index, "p_holm"] = b67.holm(g.p.values)
    T["compte"] = np.where(T["mode"] == "injecte", (T.med >= 0.05) & (T.p_holm < 0.05), T.med <= -0.05)
    return T


def main():
    T = sorted(e1.taches(), key=lambda x: 0 if x[2] in ("N", "R") else 1)
    seul = os.environ.get("VB71_SEUL")
    if seul:
        T = [x for x in T if x[2] == seul][:1]
    with Pool(int(os.environ.get("VB71_PROC", "10"))) as pool:
        res = pool.map(tache, T, chunksize=1)
    suf = f"_essai_{seul}" if seul else ""
    R = pd.DataFrame([r for rr, _ in res for r in rr])
    I = pd.DataFrame([i for _, ii in res for i in ii])
    R.to_csv(v.OUT / f"vB71_cas{suf}.csv", index=False)
    I.to_csv(v.OUT / f"vB71_capteurs{suf}.csv", index=False)
    Tv = lire(R)
    Tv.to_csv(v.OUT / f"vB71_valeurs{suf}.csv", index=False)
    pd.set_option("display.width", 250)
    piv = Tv.pivot_table(index=["mode", "eval", "categorie", "valeur"], columns="jeu", values="med").round(3)
    cpt = Tv.pivot_table(index=["mode", "eval", "categorie", "valeur"], columns="jeu", values="compte", aggfunc="first")
    cap = I.groupby(["mode", "jeu", "categorie", "estimation"])[["rappel", "nb_capteurs", "n_capteurs"]].median().round(2)
    txt = (piv.to_string() + "\n\nCompte\n" + cpt.to_string() + "\n\nCapteurs inférés (rappel des capteurs vrais)\n"
           + cap.to_string())
    (v.OUT / f"vB71{suf}.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
