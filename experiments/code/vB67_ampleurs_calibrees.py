"""vB67 — Partie 2 refaite à des ampleurs calibrées sur la dégradation (revue du 19 oct.).

Pourquoi : les injections de vB51–vB66 étaient en σ_bruit (différences entre cycles consécutifs, capteurs bruts) ;
rapporté à l'amplitude de la dégradation, 3 σ_bruit valait ×175 (C-MAPSS), ×4,5 (N-CMAPSS), ×3 (batteries),
×1,4 (roulements) : tailles non comparables entre jeux, sauts géants sur C-MAPSS (JOURNAL, revue du 19 oct.).

Unité d'injection (fixée avant de mesurer) : amplitude de dégradation du capteur j, A_j = médiane sur les unités
d'entraînement de |moyenne des 10 % derniers cycles − moyenne des 10 % premiers| (au moins 3 cycles), calculée sur les
moyennes par cycle des résidus du modèle de contexte (quadratique en W, vB62) ; ampleur f ∈ {0,25 ; 1 ; 2} :
  - mesure : un capteur décalé de ± f·A_j ;
  - maintenance : remise à zéro de vB60, puis 2 capteurs décalés de ± f·A_j ;
  - production : 2 capteurs reçoivent + f·A_j × (W standardisé · a).
Tout le reste est repris : cas, tirages, graines, méthodes, mesure (vB60) ; réponses, estimation, détection (vB62) ;
alarmes, tri et nouvelle référence avec le journal complet (vB63, vB65). Le placebo de vB62 ne dépend pas de
l'ampleur et n'est pas refait.
Lecture (fixée avant de mesurer), par jeu, catégorie, ampleur et méthode ; valeur d'un cas = moyenne de ses 2 tirages ;
médiane sur les cas, IC à 95 % par bootstrap sur les cas (2 000 tirages, graine 0), test de Wilcoxon apparié, Holm sur
toutes les comparaisons d'une même famille :
  - dégât = sans_journal − propre (référence idéale de vB60 non refaite : pour la maintenance, la vérité change) ;
  - valeur du journal = date − détection (estimation brute) ;
  - valeur de la catégorie = meilleure(cat) − meilleure(date) ; valeur des capteurs = meilleure(cat_capteurs) −
    meilleure(cat) (meilleure = maximum des deux estimations, choisi par cas) ;
  - une valeur « compte » si médiane ≥ + 0,05 et p de Holm < 0,05 ;
  - alarmes : faux épisodes par unité et détection, sans journal, tri, nouvelle référence (critères de vB63 et vB65).

Machine : Mac. Sortie : results/vB67_cas.csv, vB67_alarmes.csv, vB67.txt
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
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB51_evenements_injectes as e1  # noqa: E402
import vB60_trois_categories as b60  # noqa: E402
import vB62_journal_contexte as b62  # noqa: E402
import vB63_alarmes_journal as q4  # noqa: E402
import vB65_rereference as rr  # noqa: E402

AMPLEURS = (0.25, 1.0, 2.0)
METHODES = ["lenteur", "tendance_reg", "resid_quad", "span_W"]


def amplitude(tr, S, W):
    ctx = b62.Contexte(tr, S, W)
    fv = pd.DataFrame(ctx.residu(tr, S, W)).assign(unit=tr.unit.values, cycle=tr.cycle.values)
    fv = fv.groupby(["unit", "cycle"]).mean()
    amp = []
    for _, g in fv.groupby(level=0):
        k = max(3, len(g) // 10)
        amp.append(np.abs(g.values[-k:].mean(0) - g.values[:k].mean(0)))
    return np.median(amp, 0) + 1e-12


def injecter(df, S, W, cat, ev, unite, mu, sd):
    """unite = f·A_j / 3 : les fonctions de vB60 multiplient par 3."""
    return b60.injecter(df, S, W, cat, ev, unite, mu, sd)


def tache(args):
    jeu, cas, fonction, params = args
    tr, te, S, W = {"N": e1.cas_ncmapss, "C": e1.cas_cmapss, "B": e1.cas_batteries, "R": e1.cas_roulements}[fonction](*params)
    A = amplitude(tr, S, W)
    mu, sd = tr[W].values.mean(0), tr[W].values.std(0) + 1e-12
    rows = [dict(jeu=jeu, cas=cas, ampleur=0.0, categorie="-", tirage=-1, reponse="propre", estimation="-",
                 **e1.mesurer(tr, te, S, W))]
    alarmes = []
    for f in AMPLEURS:
        unite = f * A / 3
        for cat in b60.CATEGORIES:
            for tg in b60.TIRAGES:
                rng = np.random.default_rng(1000 * tg + b60.CATEGORIES.index(cat))
                ev_tr, ev_te = b60.tirer(tr, S, W, cat, rng), b60.tirer(te, S, W, cat, rng)
                tr_i, te_i = injecter(tr, S, W, cat, ev_tr, unite, mu, sd), injecter(te, S, W, cat, ev_te, unite, mu, sd)
                base = dict(jeu=jeu, cas=cas, ampleur=f, categorie=cat, tirage=tg)
                rows.append(dict(base, reponse="sans_journal", estimation="-", **e1.mesurer(tr_i, te_i, S, W)))
                for est in b62.ESTIMATIONS:
                    for niveau in b62.NIVEAUX:
                        a, b = b62.appliquer(tr_i, te_i, S, W, cat, ev_tr, ev_te, niveau, est)
                        rows.append(dict(base, reponse=niveau, estimation=est, **e1.mesurer(a, b, S, W)))
                ctx = b62.Contexte(tr_i, S, W)
                d_tr, d_te = b62.detecter(tr_i, S, W, ctx, "est_brut"), b62.detecter(te_i, S, W, ctx, "est_brut")
                a, b = b62.appliquer(tr_i, te_i, S, W, cat, d_tr, d_te, "date", "est_brut")
                rows.append(dict(base, reponse="detection", estimation="est_brut", **e1.mesurer(a, b, S, W)))
                vraies = b60.dates(ev_te)
                for m, sc in q4.scores(tr_i, te_i, S, W).items():
                    alarmes.append(dict(base, condition="aucun", methode=m, **q4.evaluer(sc, {})))
                    alarmes.append(dict(base, condition="tri", methode=m, **q4.evaluer(sc, vraies)))
                    alarmes.append(dict(base, condition="reref", methode=m, **rr.evaluer_reref(sc, vraies)))
    for m, sc in q4.scores(tr, te, S, W).items():
        alarmes.append(dict(jeu=jeu, cas=cas, ampleur=0.0, categorie="-", tirage=-1, condition="propre", methode=m,
                            **q4.evaluer(sc, {})))
    return rows, alarmes


def holm(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m, cur = len(idx), 0.0
    for r, i in enumerate(idx):
        cur = max(cur, min(1.0, (m - r) * p[i]))
        out[i] = cur
    return out


def resume(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return dict(n=len(x), med=np.nan, ic_bas=np.nan, ic_haut=np.nan, p=np.nan)
    bs = np.median(x[np.random.default_rng(0).integers(0, len(x), (2000, len(x)))], axis=1)
    p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    return dict(n=len(x), med=np.median(x), ic_bas=np.percentile(bs, 2.5), ic_haut=np.percentile(bs, 97.5), p=p)


def lire(R):
    L = R.melt(id_vars=["jeu", "cas", "ampleur", "categorie", "tirage", "reponse", "estimation"], value_vars=METHODES,
               var_name="methode", value_name="rho")
    propre = L[L.reponse == "propre"].set_index(["jeu", "cas", "methode"]).rho
    L = L[L.reponse != "propre"].copy()
    L["rep"] = L.reponse + "|" + L.estimation
    P = L.pivot_table(index=["jeu", "cas", "ampleur", "categorie", "tirage", "methode"], columns="rep", values="rho")
    P = P.groupby(level=["jeu", "cas", "ampleur", "categorie", "methode"]).mean()      # moyenne des 2 tirages
    best = lambda n: P[[f"{n}|est_brut", f"{n}|est_ctx"]].max(axis=1)                  # noqa: E731
    V = pd.DataFrame(index=P.index)
    V["degat"] = P["sans_journal|-"] - propre.reindex(P.index.droplevel(["ampleur", "categorie"])).values
    V["journal_vs_detection"] = P["date|est_brut"] - P["detection|est_brut"]
    V["gain_date"] = best("date") - P["sans_journal|-"]
    V["categorie_vs_date"] = best("cat") - best("date")
    V["capteurs_vs_categorie"] = best("cat_capteurs") - best("cat")
    lignes = []
    for (j, f, c, m), g in V.groupby(level=["jeu", "ampleur", "categorie", "methode"]):
        for col in V.columns:
            lignes.append(dict(jeu=j, ampleur=f, categorie=c, methode=m, valeur=col, **resume(g[col].values)))
    T = pd.DataFrame(lignes)
    T["p_holm"] = np.nan
    for col, g in T.groupby("valeur"):
        T.loc[g.index, "p_holm"] = holm(g.p.values)
    T["compte"] = (T.med >= 0.05) & (T.p_holm < 0.05)
    return T


def main():
    T = sorted(e1.taches(), key=lambda x: 0 if x[2] in ("N", "R") else 1)
    with Pool(10) as pool:
        res = pool.map(tache, T, chunksize=1)
    R = pd.DataFrame([r for rr_, _ in res for r in rr_])
    Al = pd.DataFrame([a for _, aa in res for a in aa])
    R.to_csv(v.OUT / "vB67_cas.csv", index=False)
    Al.to_csv(v.OUT / "vB67_alarmes.csv", index=False)
    Tv = lire(R)
    Tv.to_csv(v.OUT / "vB67_valeurs.csv", index=False)
    pd.set_option("display.width", 250)
    piv = Tv.pivot_table(index=["valeur", "jeu", "categorie", "methode"], columns="ampleur", values="med")
    cpt = Tv.pivot_table(index=["valeur", "jeu", "categorie", "methode"], columns="ampleur", values="compte",
                         aggfunc="first")
    a = Al.groupby(["jeu", "categorie", "ampleur", "methode", "condition"])[["faux_par_unite", "detection"]].median()
    txt = ("Valeurs (médianes sur cas)\n" + piv.round(3).to_string() + "\n\nCompte (médiane ≥ 0,05 et Holm < 0,05)\n"
           + cpt.to_string() + "\n\nAlarmes\n" + a.unstack("condition").round(3).to_string())
    (v.OUT / "vB67.txt").write_text(txt + "\n")
    print(txt[:3000])


if __name__ == "__main__":
    main()
