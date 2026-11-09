"""vB60 — E1 étendu : trois catégories de ruptures et contenu du journal (PAPIER_ICDE § 9, décision du 19 oct.).

Question : pour suivre une dégradation sans étiquettes malgré des ruptures brusques et durables, que doit contenir un
journal : rien, la date, la date et la catégorie, ou la date, la catégorie et les capteurs touchés ?

Cas, vérité, mesure et méthodes : ceux de vB51 (51 cas : N-CMAPSS 12, C-MAPSS 30, batteries 5, roulements 4 ;
Spearman par unité de test, médiane = valeur du cas ; lenteur, tendance_reg, resid_quad, span_W).
Deux événements par unité (entraînement et test), à des cycles tirés entre 10 et 90 % de sa vie, une catégorie à la
fois. Catégories (Martin, 19 oct. : les ruptures des robots sont des maintenances ou des changements de production et
ne touchent souvent que quelques axes) :
  - `mesure` (recalibration) : un capteur décalé de ± 3 σ_bruit à partir de la date (σ_bruit de vB51) ;
  - `maintenance` : remise à zéro partielle de l'usure, puis réorganisation de 2 capteurs. L'âge effectif de l'unité
    (en rang de cycle) retombe à r × l'âge atteint, r tiré uniformément dans [0 ; 0,5] ; les cycles suivants rejouent
    les lignes de l'unité à l'âge effectif correspondant (capteurs, contexte ET vérité), puis 2 capteurs tirés au
    hasard sont décalés de ± 3 σ_bruit. Référence « idéale » : la même remise à zéro sans réorganisation ;
  - `production` : 2 capteurs acquièrent une dépendance nouvelle au contexte : + 3 σ_bruit × (W standardisé · a),
    a vecteur unitaire tiré au hasard ; la santé continue.
Réponses selon le contenu du journal (le journal est exact et complet ; la qualité des dates est traitée par vB51) :
  - `sans_journal` : rien ;
  - date seule, deux réponses génériques : `date_realign` = réalignement avec détection de vB51 (version 2) ;
    `date_segment` = nouveau modèle par segment : chaque segment entre deux dates est recentré, capteur par capteur,
    sur la médiane de ses k premiers cycles puis replacé sur celle des k premiers cycles de l'unité (chaque segment
    repart d'un état de référence) ;
  - `cat` (date et catégorie, capteurs inconnus) : mesure → réalignement avec détection ; maintenance → la vie
    redémarre : les unités d'ENTRAÎNEMENT sont coupées aux dates (chaque segment devient une vie, l'ordre des cycles
    repart), les capteurs ne sont pas touchés (le saut de la remise à zéro est gardé) ; production → réajustement du
    lien contexte → capteur sur TOUS les capteurs ;
  - `cat_capteurs` (date, catégorie et capteurs) : mesure → retrait du saut (niveau, ou échelle si le rapport des
    écarts dépasse 3) sur les seuls capteurs nommés ; maintenance → coupure des vies d'entraînement ET retrait du saut
    sur les capteurs nommés ; production → réajustement sur les capteurs nommés.
  Correction de conception après un essai sur un cas roulements et un cas batteries (avant tout lancement complet) :
  dans la première version, `cat` laissait la maintenance telle quelle ; les méthodes à a priori temporel apprennent
  alors sur des vies où l'âge croît toujours et la référence idéale tombait près de 0. Savoir qu'une maintenance a eu
  lieu, c'est savoir que la vie redémarre : d'où la coupure. La référence idéale (remise à zéro sans réorganisation)
  est mesurée avec la même coupure (information complète, mesure intacte).
  Réajustement du lien contexte → capteur j à une date t : régressions linéaires de j sur [1, W] avant t (depuis la
  date précédente) et après t (jusqu'à la suivante) ; après t, j ← j − (W − moyenne) · (β_après − β_avant).
  k = max(10, 5 % des cycles de l'unité), comme vB51. Dispersions : `vB51.echelle` (correction du bug du 19 oct.,
  voir l'en-tête de vB51 ; les résultats antérieurs sur C-MAPSS sont faux).
Lecture (usage interne, fixée avant de mesurer), par jeu, catégorie et méthode, médiane sur cas et tirages :
  (1) dégât = sans_journal − référence (propre ; idéale pour la maintenance) ; la catégorie « fait échouer » si ≤ −0,05 ;
  (2) gain de chaque réponse = réponse − sans_journal ;
  (3) valeur de la catégorie = cat − max(date_realign, date_segment) ; valeur des capteurs = cat_capteurs − cat ;
      une information « compte » si sa valeur ≥ +0,05.
Attendus (déclarés) : maintenance : les réponses de date seule effacent la remise à zéro et font perdre, la
coupure des vies récupère ; production :
ni réalignement ni segment ne corrigent une pente ; mesure : la catégorie n'ajoute rien sans les capteurs.

Machine : Mac. Sortie : results/vB60_cas.csv, vB60.txt
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

CATEGORIES = ["mesure", "maintenance", "production"]
REPONSES = ["sans_journal", "date_realign", "date_segment", "cat", "cat_capteurs"]
N_EV, TIRAGES, N_CAPT = 2, (0, 1), 2


# --------------------------------------------------------------------------------------------------
def tirer(df, S, W, cat, rng):
    """Par unité : liste d'événements {t, capteurs, signes, r, a}."""
    ev = {}
    for u, g in df.groupby("unit"):
        cy = np.sort(g.cycle.unique())
        lo, hi = int(0.1 * len(cy)), max(int(0.9 * len(cy)) - 1, int(0.1 * len(cy)) + 1)
        dates = np.sort(rng.choice(cy[lo:hi + 1], N_EV, replace=False))
        n = 1 if cat == "mesure" else N_CAPT
        ev[u] = [dict(t=int(t), capt=[int(x) for x in rng.choice(len(S), n, replace=False)],
                      sg=[int(x) for x in rng.choice([-1, 1], n)], r=float(rng.uniform(0, 0.5)),
                      a=[(lambda z: z / np.linalg.norm(z))(rng.normal(size=len(W))) for _ in range(n)])
                 for t in dates]
    return ev


def rejouer(df, ev):
    """Maintenance : l'âge effectif (rang de cycle) retombe à r × âge à chaque date ; les lignes sont rejouées."""
    parts = []
    for u, g in df.groupby("unit", sort=False):
        g = g.sort_values("cycle")
        cy = np.sort(g.cycle.unique())
        age = np.arange(len(cy), dtype=float)
        for e in sorted(ev.get(u, []), key=lambda x: x["t"]):
            p = int(np.searchsorted(cy, e["t"]))
            age[p:] = e["r"] * age[p] + np.arange(len(cy) - p)
        src = np.clip(np.round(age).astype(int), 0, len(cy) - 1)
        idx = g.groupby("cycle").indices
        blocs = [idx[cy[s]] for s in src]
        h = g.iloc[np.concatenate(blocs)].copy()
        h["cycle"] = np.repeat(cy, [len(b) for b in blocs])
        parts.append(h)
    return pd.concat(parts, ignore_index=True)


def decaler(df, S, ev, sig):
    df = df.copy()
    X = df[S].values.astype(float)
    for u, liste in ev.items():
        m_u = df.unit.values == u
        for e in liste:
            m = m_u & (df.cycle.values >= e["t"])
            for j, sg in zip(e["capt"], e["sg"]):
                X[m, j] += sg * 3 * sig[j]
    df[S] = X
    return df


def produire(df, S, W, ev, sig, mu, sd):
    df = df.copy()
    X = df[S].values.astype(float)
    Z = (df[W].values - mu) / sd
    for u, liste in ev.items():
        m_u = df.unit.values == u
        for e in liste:
            m = m_u & (df.cycle.values >= e["t"])
            for j, a in zip(e["capt"], e["a"]):
                X[m, j] += 3 * sig[j] * (Z[m] @ a)
    df[S] = X
    return df


def injecter(df, S, W, cat, ev, sig, mu, sd, ideal=False):
    if cat == "mesure":
        return decaler(df, S, ev, sig)
    if cat == "maintenance":
        d = rejouer(df, ev)
        return d if ideal else decaler(d, S, ev, sig)
    return produire(df, S, W, ev, sig, mu, sd)


# --------------------------------------------------------------------------------------------------
def dates(ev):
    return {u: sorted(e["t"] for e in liste) for u, liste in ev.items()}


def segmenter(df, S, ds):
    """Nouveau modèle par segment : chaque segment repart du niveau des k premiers cycles de l'unité."""
    df = df.copy()
    X = df[S].values.astype(float)
    unit, cyc = df.unit.values, df.cycle.values
    for u, dd in ds.items():
        m_u = unit == u
        cy = np.unique(cyc[m_u])
        k = e1.fenetre(cy)
        ref = np.median(X[m_u & (cyc < cy[0] + k)], axis=0)
        bornes = list(dd) + [np.inf]
        for t0, t1 in zip(bornes[:-1], bornes[1:]):
            seg = m_u & (cyc >= t0) & (cyc < t1)
            debut = seg & (cyc < t0 + k)
            if debut.sum() >= 3:
                X[seg] += ref - np.median(X[debut], axis=0)
    df[S] = X
    return df


def retirer_saut(df, S, ev):
    """Capteurs nommés : retrait du saut à la date (échelle si le rapport des écarts dépasse 3, sinon niveau)."""
    df = df.copy()
    X = df[S].values.astype(float)
    unit, cyc = df.unit.values, df.cycle.values
    for u, liste in ev.items():
        m_u = unit == u
        k = e1.fenetre(np.unique(cyc[m_u]))
        for e in liste:
            t = e["t"]
            av, ap, post = m_u & (cyc >= t - k) & (cyc < t), m_u & (cyc >= t) & (cyc < t + k), m_u & (cyc >= t)
            if av.sum() < 3 or ap.sum() < 3:
                continue
            for j in e["capt"]:
                a, p = X[av, j], X[ap, j]
                ma, mp = np.median(a), np.median(p)
                sa, sp = e1.echelle(a), e1.echelle(p)
                if sa > 0 and sp > 0 and abs(np.log(sp / sa)) > np.log(3):
                    X[post, j] = (X[post, j] - mp) * (sa / sp) + ma
                else:
                    X[post, j] += ma - mp
    df[S] = X
    return df


def reajuster(df, S, W, ev, tous):
    """Réajustement du lien contexte → capteur après chaque date (capteurs nommés, ou tous)."""
    df = df.copy()
    X = df[S].values.astype(float)
    Wv = df[W].values.astype(float)
    unit, cyc = df.unit.values, df.cycle.values
    for u, liste in ev.items():
        m_u = unit == u
        ts = sorted(e["t"] for e in liste)
        bornes = [-np.inf] + ts + [np.inf]
        for i, e in enumerate(sorted(liste, key=lambda x: x["t"])):
            av = m_u & (cyc >= bornes[i]) & (cyc < bornes[i + 1])
            ap = m_u & (cyc >= bornes[i + 1]) & (cyc < bornes[i + 2])
            if av.sum() < len(W) + 3 or ap.sum() < len(W) + 3:
                continue
            Aa = np.column_stack([np.ones(av.sum()), Wv[av]])
            Ap = np.column_stack([np.ones(ap.sum()), Wv[ap]])
            for j in (range(len(S)) if tous else e["capt"]):
                ba = np.linalg.lstsq(Aa, X[av, j], rcond=None)[0][1:]
                bp = np.linalg.lstsq(Ap, X[ap, j], rcond=None)[0][1:]
                X[ap, j] -= (Wv[ap] - Wv[ap].mean(0)) @ (bp - ba)
    df[S] = X
    return df


def couper(df, ev):
    """Maintenance connue : chaque segment entre deux dates devient une vie d'entraînement."""
    df = df.copy()
    seg = np.zeros(len(df), int)
    for u, liste in ev.items():
        m_u = df.unit.values == u
        for e in liste:
            seg[m_u & (df.cycle.values >= e["t"])] += 1
    df["unit"] = df.unit.astype(str) + "#" + seg.astype(str)
    return df


def repondre(df, S, W, cat, ev, rep, entr=False):
    if cat == "maintenance" and rep in ("cat", "cat_capteurs"):
        d = retirer_saut(df, S, ev) if rep == "cat_capteurs" else df
        return couper(d, ev) if entr else d
    if rep == "date_realign":
        return e1.realigner(df, S, dates(ev))
    if rep == "date_segment":
        return segmenter(df, S, dates(ev))
    if rep == "cat":
        return e1.realigner(df, S, dates(ev)) if cat == "mesure" else reajuster(df, S, W, ev, tous=True)
    if rep == "cat_capteurs":
        return reajuster(df, S, W, ev, tous=False) if cat == "production" else retirer_saut(df, S, ev)
    return df


# --------------------------------------------------------------------------------------------------
def tache(args):
    jeu, cas, fonction, params = args
    tr, te, S, W = {"N": e1.cas_ncmapss, "C": e1.cas_cmapss, "B": e1.cas_batteries, "R": e1.cas_roulements}[fonction](*params)
    sig = e1.sigma_bruit(tr, S)
    mu, sd = tr[W].values.mean(0), tr[W].values.std(0) + 1e-12
    rows = [dict(jeu=jeu, cas=cas, categorie="-", tirage=-1, reponse="propre", **e1.mesurer(tr, te, S, W))]
    for cat in CATEGORIES:
        for tg in TIRAGES:
            rng = np.random.default_rng(1000 * tg + CATEGORIES.index(cat))
            ev_tr, ev_te = tirer(tr, S, W, cat, rng), tirer(te, S, W, cat, rng)
            tr_i, te_i = injecter(tr, S, W, cat, ev_tr, sig, mu, sd), injecter(te, S, W, cat, ev_te, sig, mu, sd)
            base = dict(jeu=jeu, cas=cas, categorie=cat, tirage=tg)
            if cat == "maintenance":
                rows.append(dict(base, reponse="ideal", **e1.mesurer(
                    couper(injecter(tr, S, W, cat, ev_tr, sig, mu, sd, ideal=True), ev_tr),
                    injecter(te, S, W, cat, ev_te, sig, mu, sd, ideal=True), S, W)))
            deja = {}
            for rep in REPONSES:
                cle = "date_realign" if (rep == "cat" and cat == "mesure") else rep
                if cle not in deja:
                    deja[cle] = e1.mesurer(repondre(tr_i, S, W, cat, ev_tr, cle, entr=True),
                                           repondre(te_i, S, W, cat, ev_te, cle), S, W)
                rows.append(dict(base, reponse=rep, **deja[cle]))
    if e1.ERREURS:
        print(f"{jeu} {cas} : {len(e1.ERREURS)} échecs, ex. {sorted(set(e1.ERREURS))[:3]}", flush=True)
    return rows


def lire(R):
    meths = ["lenteur", "tendance_reg", "resid_quad", "span_W"]
    L = R.melt(id_vars=["jeu", "cas", "categorie", "tirage", "reponse"], value_vars=meths, var_name="methode",
               value_name="rho")
    propre = L[L.reponse == "propre"].set_index(["jeu", "cas", "methode"]).rho
    L = L[L.reponse != "propre"]
    P = L.pivot_table(index=["jeu", "cas", "categorie", "tirage", "methode"], columns="reponse", values="rho")
    P["reference"] = P["ideal"].where(P.index.get_level_values("categorie") == "maintenance",
                                      propre.reindex(P.index.droplevel(["categorie", "tirage"])).values)
    P["degat"] = P.sans_journal - P.reference
    for r in REPONSES[1:]:
        P[f"gain_{r}"] = P[r] - P.sans_journal
    P["valeur_categorie"] = P.cat - P[["date_realign", "date_segment"]].max(axis=1)
    P["valeur_capteurs"] = P.cat_capteurs - P.cat
    cols = ["reference", "sans_journal", "degat"] + [f"gain_{r}" for r in REPONSES[1:]] + ["valeur_categorie",
                                                                                            "valeur_capteurs"]
    return P.groupby(["jeu", "categorie", "methode"])[cols].median()


def main():
    T = e1.taches()
    seul = os.environ.get("VB60_SEUL")
    if seul:
        T = [x for x in T if x[2] == seul][:1]
    T = sorted(T, key=lambda x: 0 if x[2] in ("N", "R") else 1)
    with Pool(int(os.environ.get("VB60_PROC", "10"))) as pool:
        R = pd.DataFrame([r for rr in pool.map(tache, T, chunksize=1) for r in rr])
    suf = f"_essai_{seul}" if seul else ""
    R.to_csv(v.OUT / f"vB60_cas{suf}.csv", index=False)
    pd.set_option("display.width", 250)
    txt = lire(R).round(3).to_string()
    print(txt)
    (v.OUT / f"vB60{suf}.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
