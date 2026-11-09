"""vB63 — Q4 : le journal en production, pour expliquer les alarmes (Martin, 19 oct.).

Question : quand des ruptures non corrigées déclenchent des alarmes, quelle part de ces fausses alarmes un journal
permet-il d'expliquer, et combien de vraies alarmes masque-t-il ? Le journal ne touche pas aux méthodes : il sert à
trier les alarmes.

Cas, catégories, injections et tirages : vB60 (mêmes graines). Méthodes : les 4 de vB60, ajustées sur l'entraînement
(sans correction). Score par cycle = moyenne des fenêtres du cycle, unités de test, lignes évaluées.
Alarmes (fixées avant de mesurer) :
  - référence de chaque unité de test = ses 10 % premiers cycles (au moins 5 ; les événements tombent après 10 %) ;
    z = (score − médiane) / (1,4826 × écart absolu médian) de cette référence ;
  - épisode d'alarme = au moins 3 cycles consécutifs à z > 3 ; deux épisodes séparés de ≤ 2 cycles sont fusionnés ;
  - un épisode est FAUX s'il commence quand la vérité de l'unité (après injection) est sous 50 % de son étendue ;
    une unité est DÉTECTÉE si une alarme est active à un cycle où la vérité dépasse 70 % ; ses vraies alarmes sont
    MASQUÉES si tous les épisodes actifs à ces cycles sont expliqués par le journal. (Correction de définition après
    un essai sur un cas C-MAPSS et un cas N-CMAPSS, avant le lancement complet : dans la première version un épisode
    n'était vrai que s'il COMMENÇAIT au-delà de 70 % ; une dégradation progressive donne un seul long épisode qui
    commence avant, et la détection valait 0.)
  - un épisode est EXPLIQUÉ par le journal s'il commence dans les k cycles qui suivent une date journalisée
    (k = max(10, 5 % des cycles de l'unité), comme vB51).
Journaux : complet exact ; complétude 50 % ; dates fausses de ± 5 cycles (tirages de vB51) ; placebo sur données
propres (journal tiré comme dans vB60, aucune rupture) pour mesurer le masquage à vide.
Mesures par cas et méthode : faux épisodes par unité ; part des faux épisodes expliqués ; part des unités avec au moins
un vrai épisode ; part des unités dont tous les vrais épisodes sont expliqués (vraies alarmes masquées).
Lecture (usage interne, fixée avant de mesurer), médianes par jeu, catégorie et méthode :
  (1) les ruptures « créent » des fausses alarmes si faux épisodes par unité (injecté − propre) ≥ 0,2 ;
  (2) le journal « explique » si part expliquée ≥ 0,5 avec un masquage ≤ 0,05 ;
  (3) qualité : même lecture pour complétude 50 % et dates ± 5.

Machine : Mac. Sortie : results/vB63_cas.csv, vB63.txt
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

METHODES = ["lenteur", "tendance_reg", "resid_quad", "span_W"]
TAU, L_MIN, TROU, REF = 3.0, 3, 2, 0.10
JOURNAUX = ["journal", "complet50", "date5"]


def scores(tr, te, S, W):
    """Score par cycle de chaque méthode sur les unités de test (lignes évaluées) ; méthode en échec → absente."""
    import vB32_tendance_contre_lenteur as t32
    import vB42_quoi_journaliser as b42
    tev = te[te["eval"]]
    out = {}
    try:
        t = t32.methodes(tr[S].values, tr.unit.values, tr.cycle.values)
        fs = {"lenteur": lambda S_, W_: t["lenteur"](S_), "tendance_reg": lambda S_, W_: t["tendance_reg"](S_)}
    except Exception as e:                                       # noqa: BLE001
        e1.ERREURS.append(f"tendances: {type(e).__name__}")
        fs = {}
    try:
        fs.update({k: f for k, f in b42.lineaires(tr[S].values, tr[W].values, tr.unit.values, tr.cycle.values).items()
                   if k in METHODES})
    except Exception as e:                                       # noqa: BLE001
        e1.ERREURS.append(f"lineaires: {type(e).__name__}")
    for k, f in fs.items():
        try:
            sc = f(tev[S].values, tev[W].values)
            if np.all(np.isfinite(sc)):
                out[k] = tev.assign(s=sc).groupby(["unit", "cycle"]).agg(s=("s", "mean"), v=("verite", "first")).reset_index()
        except Exception as e:                                   # noqa: BLE001
            e1.ERREURS.append(f"{k}: {type(e).__name__}")
    return out


def episodes(z):
    """Épisodes (début, fin exclue) : ≥ L_MIN cycles consécutifs au-dessus de TAU, trous ≤ TROU fusionnés."""
    haut = np.concatenate([[False], z > TAU, [False]])
    d = np.diff(haut.astype(int))
    runs = [(a, b) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]) if b - a >= L_MIN]
    ep = []
    for a, b in runs:
        if ep and a - ep[-1][1] <= TROU:
            ep[-1][1] = int(b)
        else:
            ep.append([int(a), int(b)])
    return ep


def evaluer(sc, dates):
    """Mesures d'alarme pour une méthode : sc (unit, cycle, s, v), dates = {unité: [dates journalisées]}."""
    faux = expl = 0
    n_u = det = masque = 0
    for u, g in sc.groupby("unit"):
        g = g.sort_values("cycle")
        c, s, vv = g.cycle.values, g.s.values, g.v.values
        nref = max(5, int(REF * len(c)))
        if len(c) < nref + L_MIN:
            continue
        ref = s[:nref]
        z = (s - np.median(ref)) / (1.4826 * np.median(np.abs(ref - np.median(ref))) + 1e-12)
        lo, hi = vv.min(), vv.max()
        rel = (vv - lo) / (hi - lo + 1e-12)
        k = e1.fenetre(c)
        dd = np.array(dates.get(u, []))
        n_u += 1
        vrais, vrais_expl = 0, 0
        for i, j in episodes(z):
            ex = bool(len(dd) and np.any((c[i] >= dd) & (c[i] < dd + k)))
            if rel[i] < 0.5:
                faux += 1
                expl += ex
            if np.any(rel[i:j] > 0.7):                          # alarme active en fin de vie
                vrais += 1
                vrais_expl += ex
        det += vrais > 0
        masque += vrais > 0 and vrais_expl == vrais
    return dict(unites=n_u, faux_par_unite=faux / max(n_u, 1), part_expliquee=expl / faux if faux else np.nan,
                detection=det / max(n_u, 1), masquage=masque / max(n_u, 1))


def tache(args):
    jeu, cas, fonction, params = args
    tr, te, S, W = {"N": e1.cas_ncmapss, "C": e1.cas_cmapss, "B": e1.cas_batteries, "R": e1.cas_roulements}[fonction](*params)
    sig = e1.sigma_bruit(tr, S)
    mu, sd = tr[W].values.mean(0), tr[W].values.std(0) + 1e-12
    cycles = {u: np.sort(g.cycle.unique()) for u, g in pd.concat([tr, te]).groupby("unit")}
    rows = []
    sp = scores(tr, te, S, W)
    for m, sc in sp.items():
        rows.append(dict(jeu=jeu, cas=cas, mode="propre", categorie="-", tirage=-1, journal="aucun", methode=m,
                         **evaluer(sc, {})))
    for cat in b60.CATEGORIES:
        for tg in b60.TIRAGES:
            rng = np.random.default_rng(1000 * tg + b60.CATEGORIES.index(cat))
            ev_tr, ev_te = b60.tirer(tr, S, W, cat, rng), b60.tirer(te, S, W, cat, rng)
            vraies = b60.dates(ev_te)
            si = scores(b60.injecter(tr, S, W, cat, ev_tr, sig, mu, sd), b60.injecter(te, S, W, cat, ev_te, sig, mu, sd),
                        S, W)
            base = dict(jeu=jeu, cas=cas, categorie=cat, tirage=tg)
            for m, sc in si.items():
                rows.append(dict(base, mode="injecte", journal="aucun", methode=m, **evaluer(sc, {})))
                for jn in JOURNAUX:
                    rj = np.random.default_rng(7 + 1000 * tg + e1.CONDITIONS.index(jn))
                    lg = e1.journal({u: [(t, 0, 0, 0) for t in d] for u, d in vraies.items()}, jn, cycles, rj)
                    rows.append(dict(base, mode="injecte", journal=jn, methode=m, **evaluer(sc, lg)))
            for m, sc in sp.items():                                # placebo : données propres, journal tiré
                rows.append(dict(base, mode="placebo", journal="journal", methode=m, **evaluer(sc, vraies)))
    return rows


def main():
    T = sorted(e1.taches(), key=lambda x: 0 if x[2] in ("N", "R") else 1)
    seul = os.environ.get("VB63_SEUL")
    if seul:
        T = [x for x in T if x[2] == seul][:1]
    with Pool(int(os.environ.get("VB63_PROC", "10"))) as pool:
        R = pd.DataFrame([r for rr in pool.map(tache, T, chunksize=1) for r in rr])
    suf = f"_essai_{seul}" if seul else ""
    R.to_csv(v.OUT / f"vB63_cas{suf}.csv", index=False)
    cols = ["faux_par_unite", "part_expliquee", "detection", "masquage"]
    t = R.groupby(["jeu", "categorie", "mode", "journal", "methode"])[cols].median().unstack(["mode", "journal"])
    pd.set_option("display.width", 320)
    txt = t.round(3).to_string()
    print(txt)
    (v.OUT / f"vB63{suf}.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
