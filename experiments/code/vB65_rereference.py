"""vB65 — Q4 bis : après un événement journalisé, repartir d'une nouvelle référence d'alarme.

Pourquoi (vB63) : expliquer un épisode d'alarme par la date d'un événement masque les vraies alarmes de 27 à 41 % des
unités de C-MAPSS, parce qu'un saut persistant et la dégradation forment un seul épisode (91 % des masquages).
Question : si, après chaque date journalisée, la référence de l'alarme est réapprise (comme la « nouvelle normale »
de la surveillance v2 du papier Robots), le masquage disparaît-il sans perte de détection ?

Tout est repris de vB63 (cas, catégories, injections, tirages, méthodes, scores, épisodes, vérité, seuil 3, k).
Nouvelle référence : pour chaque date journalisée t d'une unité, les cycles [t, t + k) ne déclenchent pas d'alarme et
servent de référence (médiane, 1,4826 × écart absolu médian) aux cycles ≥ t + k, jusqu'à la date suivante ; avant la
première date, la référence de vB63 (10 % premiers cycles).
Conditions : aucun journal ; tri (vB63) ; nouvelle référence avec journal complet, complétude 50 %, dates ± 5 ;
placebo (données propres, journal tiré comme vB60) avec nouvelle référence.
Mesures : faux épisodes par unité, détection (alarme active à un cycle où la vérité dépasse 70 %).
Lecture (usage interne, fixée avant de mesurer), médianes par jeu, catégorie et méthode : la nouvelle référence
« supprime le coût » si faux épisodes ≤ propre + 0,05 et détection ≥ détection sans journal − 0,05 ; son coût à vide
= détection placebo − détection propre.

Machine : Mac. Sortie : results/vB65_cas.csv, vB65.txt
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
import vB63_alarmes_journal as q4  # noqa: E402


def robuste(x):
    return np.median(x), 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12


def evaluer_reref(sc, dates):
    faux = det = n_u = 0
    for u, g in sc.groupby("unit"):
        g = g.sort_values("cycle")
        c, s, vv = g.cycle.values, g.s.values, g.v.values
        nref = max(5, int(q4.REF * len(c)))
        if len(c) < nref + q4.L_MIN:
            continue
        k = e1.fenetre(c)
        z = np.full(len(c), -np.inf)
        m0, s0 = robuste(s[:nref])
        bornes = sorted(t for t in dates.get(u, []) if t > c[nref - 1])
        debut_seg = [c[0]] + bornes
        fins = bornes + [np.inf]
        for i_seg, (a, b) in enumerate(zip(debut_seg, fins)):
            seg = (c >= a) & (c < b)
            if i_seg == 0:
                z[seg] = (s[seg] - m0) / s0
                continue
            ref = seg & (c < a + k)
            if ref.sum() < 3:
                z[seg] = (s[seg] - m0) / s0
                continue
            m, sd = robuste(s[ref])
            apres = seg & (c >= a + k)
            z[apres] = (s[apres] - m) / sd
        lo, hi = vv.min(), vv.max()
        rel = (vv - lo) / (hi - lo + 1e-12)
        n_u += 1
        ep = q4.episodes(z)
        faux += sum(rel[i] < 0.5 for i, j in ep)
        det += any(np.any(rel[i:j] > 0.7) for i, j in ep)
    return dict(unites=n_u, faux_par_unite=faux / max(n_u, 1), detection=det / max(n_u, 1))


def tache(args):
    jeu, cas, fonction, params = args
    tr, te, S, W = {"N": e1.cas_ncmapss, "C": e1.cas_cmapss, "B": e1.cas_batteries, "R": e1.cas_roulements}[fonction](*params)
    sig = e1.sigma_bruit(tr, S)
    mu, sd = tr[W].values.mean(0), tr[W].values.std(0) + 1e-12
    cycles = {u: np.sort(g.cycle.unique()) for u, g in pd.concat([tr, te]).groupby("unit")}
    rows = []
    sp = q4.scores(tr, te, S, W)
    for m, sc in sp.items():
        rows.append(dict(jeu=jeu, cas=cas, categorie="-", tirage=-1, condition="propre", methode=m,
                         **q4.evaluer(sc, {})))
    for cat in b60.CATEGORIES:
        for tg in b60.TIRAGES:
            rng = np.random.default_rng(1000 * tg + b60.CATEGORIES.index(cat))
            ev_tr, ev_te = b60.tirer(tr, S, W, cat, rng), b60.tirer(te, S, W, cat, rng)
            vraies = b60.dates(ev_te)
            si = q4.scores(b60.injecter(tr, S, W, cat, ev_tr, sig, mu, sd), b60.injecter(te, S, W, cat, ev_te, sig, mu, sd),
                           S, W)
            base = dict(jeu=jeu, cas=cas, categorie=cat, tirage=tg)
            for m, sc in si.items():
                rows.append(dict(base, condition="aucun", methode=m, **q4.evaluer(sc, {})))
                rows.append(dict(base, condition="tri", methode=m, **q4.evaluer(sc, vraies)))
                for jn in q4.JOURNAUX:
                    rj = np.random.default_rng(7 + 1000 * tg + e1.CONDITIONS.index(jn))
                    lg = e1.journal({u: [(t, 0, 0, 0) for t in d] for u, d in vraies.items()}, jn, cycles, rj)
                    rows.append(dict(base, condition=f"reref_{jn}", methode=m, **evaluer_reref(sc, lg)))
            for m, sc in sp.items():
                rows.append(dict(base, condition="placebo_reref", methode=m, **evaluer_reref(sc, vraies)))
    return rows


def main():
    T = sorted(e1.taches(), key=lambda x: 0 if x[2] in ("N", "R") else 1)
    with Pool(10) as pool:
        R = pd.DataFrame([r for rr in pool.map(tache, T, chunksize=1) for r in rr])
    R.to_csv(v.OUT / "vB65_cas.csv", index=False)
    pr = R[R.condition == "propre"].groupby(["jeu", "methode"])[["faux_par_unite", "detection"]].median()
    t = R[R.condition != "propre"].groupby(["jeu", "categorie", "methode", "condition"])[
        ["faux_par_unite", "detection"]].median().unstack("condition")
    pd.set_option("display.width", 320)
    txt = "propre\n" + pr.round(3).to_string() + "\n\n" + t.round(3).to_string()
    print(txt)
    (v.OUT / "vB65.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
