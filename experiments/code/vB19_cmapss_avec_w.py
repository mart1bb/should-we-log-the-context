"""vB19 — C-MAPSS FD002/FD004 avec W : retirer les conditions par un modèle nuit-il hors support ?

Deuxième jeu de données pour les revendications 2 et 3 de la voie B (papier B). Même expérience et mêmes
découpages que la voie A (a6, importé en lecture ; CANAL M-19, M-26 (B)) : un régime d'opération retiré
de l'entraînement — R42 (42 kft, hors support en altitude), R60 (manette 60, hors support en manette),
R20 (20 kft, dans le support des conditions mais hors de l'espace qu'elles engendrent, a6) — et les
contrôles C0 (tous les régimes vus, même régime cible). 50 % des unités en entraînement, 5 tirages.
W = les 3 réglages d'opération (s1, s2, s3), observés à chaque cycle. Pas de vérité de dégradation dans
C-MAPSS : l'âge (cycle) sert de proxy, comme en a6.

Méthodes, déclarées avant mesure — la même SFA commune (`sfa.ajuster`) appliquée à des entrées
différentes, pour isoler l'effet du retrait des conditions :
  - `sfa_brut`        : capteurs bruts (= la SFA sans W de la voie A, a6 : repère, cité) ;
  - `sfa_resid_lin`   : résidu de capteurs ~ W linéaire (ajusté sur les cycles d'entraînement) ;
  - `sfa_resid_quad`  : résidu de capteurs ~ φ(W) (W, W², produits) ;
  - `sfa_norm_regime` : capteurs standardisés par régime (moyenne et écart-type du régime sur
                        l'entraînement) — la pratique courante sur C-MAPSS ; un régime absent de
                        l'entraînement prend les statistiques du régime le plus proche en W (standardisé) ;
  - `span_W`          : SFA hors de l'espace des effets des conditions estimé avec W sur les différences
                        entre cycles consécutifs (vB08), rien n'est soustrait.
Mesures par unité de test (comme a6) : Spearman(score, cycle) sur les cycles du régime cible et sur les
autres ; décalage du niveau sain du régime cible (20 % premiers cycles) en bruits ; fausses alarmes.
Diagnostics sans étiquettes sur les unités d'entraînement : R̂ (vB14) et D (vB15), SFA brute.

Lecture fixée AVANT de mesurer (médianes sur les unités de test et les tirages) :
  - revendication 2 reproduite si, sur R42 et R60 (hors support), au moins une des SFA sur résidu perd
    ≥ 0,1 de Spearman sur le régime cible par rapport à `sfa_brut`, alors qu'en C0 toutes sont à ± 0,05 ;
    `sfa_norm_regime` attendue en échec sur le régime retiré (statistiques empruntées) ;
  - `span_W` à ± 0,05 de `sfa_brut` en C0 et sur les régimes retirés (sans dérive d'usage, W ne doit ni
    aider ni coûter : carte de vB12) ;
  - D bas (< 0,2) et R̂ élevé attendus : C-MAPSS du côté « la lenteur suffit ».

Machine : Mac, secondes à minutes. Sortie : voie_B_avec_exo/results/vB19_*.csv, vB19.txt
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
sys.path.insert(0, str(ICI.parents[1] / "code"))  # a6 (C-MAPSS) est dans code/
import vB01_sonde_marge as v  # noqa: E402  (ajoute aussi ../code au chemin)
import vB08_espace_conditions as v8  # noqa: E402
import vB14_r_sans_etiquettes as r14  # noqa: E402
import vB15_detecteur_piege as d15  # noqa: E402
import a6_cmapss_regime_retire as a6  # noqa: E402  (voie A, lecture seule)
import sfa  # noqa: E402

W_C = ["s1", "s2", "s3"]
N_TIRAGES, DEBUT = a6.N_TIRAGES, a6.DEBUT


def phi(W):
    prod = np.stack([W[:, i] * W[:, j] for i, j in combinations(range(W.shape[1]), 2)], 1)
    return np.hstack([W, W ** 2, prod])


def entrees(tr, te, X, W_C):
    """Entrées (entraînement, test) de chaque méthode fondée sur un retrait des conditions."""
    Str, Ste = tr[X].values.astype(float), te[X].values.astype(float)
    muW, sdW = tr[W_C].values.mean(0), tr[W_C].values.std(0) + 1e-12
    Wtr, Wte = (tr[W_C].values - muW) / sdW, (te[W_C].values - muW) / sdW
    out = {"sfa_brut": (Str, Ste)}
    f = LinearRegression().fit(Wtr, Str)
    out["sfa_resid_lin"] = (Str - f.predict(Wtr), Ste - f.predict(Wte))
    q = LinearRegression().fit(phi(Wtr), Str)
    out["sfa_resid_quad"] = (Str - q.predict(phi(Wtr)), Ste - q.predict(phi(Wte)))
    stats = tr.groupby("regime")[X].agg(["mean", "std"])
    reg_W = tr.groupby("regime")[W_C].mean()
    reg_Wz = (reg_W.values - muW) / sdW

    moy = stats.xs("mean", axis=1, level=1)[X]
    ect = stats.xs("std", axis=1, level=1)[X] + 1e-12

    def norm(d, Wz):
        # régime connu : ses statistiques ; régime absent de l'entraînement : régime connu le plus proche en W
        proche = reg_W.index[np.argmin(((Wz[:, None, :] - reg_Wz[None, :, :]) ** 2).sum(2), axis=1)]
        r = np.where(d.regime.isin(stats.index).values, d.regime.values, proche)
        return (d[X].values - moy.loc[r].values) / ect.loc[r].values
    out["sfa_norm_regime"] = (norm(tr, Wtr), norm(te, Wte))
    return out


def mesures_unite(t, meth, cib):
    lignes = []
    for u, g in t.groupby("unit"):
        gc, gv = g[g.regime == cib], g[g.regime != cib]
        gvd, gcd = gv[gv.frac < DEBUT], gc[gc.frac < DEBUT]
        if len(gc) < 10 or len(gcd) < 3 or len(gvd) < 10:
            continue
        bruit = gvd[meth].std()
        lignes.append(dict(unite=u, sp_cible=spearmanr(gc.cycle, gc[meth]).statistic,
                           sp_vus=spearmanr(gv.cycle, gv[meth]).statistic,
                           decalage_z=(gcd[meth].median() - gvd[meth].median()) / bruit,
                           FA=float((gcd[meth] > np.quantile(gvd[meth], 0.95)).mean())))
    return lignes


def main():
    lignes, diag = [], []
    for fd in ["FD002", "FD004"]:
        d = a6.charger(fd)
        for cas, retire, cible in a6.CAS:
            for tirage in range(N_TIRAGES):
                rng = np.random.default_rng(tirage)                    # même découpage que a6
                unites = d.unit.unique()
                tr_u = set(rng.choice(unites, size=len(unites) // 2, replace=False))
                tr = d[d.unit.isin(tr_u)]
                if retire is not None:
                    tr = tr[tr.regime != retire]
                te = d[~d.unit.isin(tr_u)]
                X = [c for c in a6.COLS[5:] if tr[c].std() > 1e-8]
                # variables d'opération constantes à l'entraînement retirées (R60 : manette seule à 100)
                Wk = [c for c in W_C if tr[c].std() > 1e-8]
                scores = {}
                for nom, (Etr, Ete) in entrees(tr, te, X, Wk).items():
                    # colonnes sans variance à l'entraînement retirées (capteurs entièrement expliqués
                    # par le régime : résidu nul ; même règle que a6 sur les capteurs bruts)
                    sd = Etr.std(0)
                    garde = sd > 1e-6 * sd.max()
                    Etr, Ete = Etr[:, garde], Ete[:, garde]
                    m = sfa.ajuster(Etr, tr.unit.values, tr.cycle.values)
                    scores[nom] = m.score(Ete)
                    if nom == "sfa_brut":
                        s_tr = m.score(Etr)
                        diag.append(dict(FD=fd, cas=cas, tirage=tirage,
                                         R_chapeau=r14.r_chapeau(s_tr, tr.unit.values, tr.cycle.values),
                                         D=d15.detecteur(s_tr, tr[Wk].values.astype(float), tr.unit.values,
                                                         tr.cycle.values)))
                ms, _ = v8.span_W(tr[X].values.astype(float), tr[Wk].values.astype(float),
                                  tr.unit.values, tr.cycle.values)
                scores["span_W"] = ms.score(te[X].values.astype(float))
                t = te.assign(**scores)
                for meth in scores:
                    for r in mesures_unite(t, meth, cible):
                        lignes.append(dict(FD=fd, cas=cas, tirage=tirage, methode=meth, **r))
        print(fd, "fait", flush=True)
    res, dg = pd.DataFrame(lignes), pd.DataFrame(diag)
    res.to_csv(v.OUT / "vB19_cmapss.csv", index=False)
    dg.to_csv(v.OUT / "vB19_diagnostics.csv", index=False)
    pd.set_option("display.width", 250)
    ordre = ["sfa_brut", "sfa_resid_lin", "sfa_resid_quad", "sfa_norm_regime", "span_W"]
    for mtr in ["sp_cible", "sp_vus", "decalage_z", "FA"]:
        print(f"\n=== {mtr} — médiane (unités de test × tirages) ===")
        p = res.groupby(["FD", "cas", "methode"])[mtr].median().unstack("methode")[ordre]
        print(p.round(2).to_string())
    print("\n=== diagnostics sans étiquettes (SFA brute, unités d'entraînement) — médianes ===")
    print(dg.groupby(["FD", "cas"])[["R_chapeau", "D"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
