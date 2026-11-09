"""vB24 — Les indicateurs sans étiquettes prédisent-ils la méthode qui gagne ?

Indicateurs, calculés sur les unités d'ENTRAÎNEMENT, sans étiquettes :
  - p_hors : part de la signature de dégradation hors de l'espace des conditions (vB23, `tendance`) ;
  - R̂ : rapport signal/bruit le long de la direction de la SFA sans W (vB14) ;
  - D : part des différences du score SFA sans W expliquée par celles des conditions (vB15).
Méthodes candidates (même SFA en aval) : `lenteur` = SFA sans W (méthode de la voie A, citée) ;
`modele` = SFA sur le résidu d'un modèle de condition quadratique en W (vB19/vB22) ; `span_W` = v2 (DB-08).

Règle, fixée AVANT de mesurer :
  1. si p_hors < 0,3 → `modele` (la signature est dans l'espace des conditions : il faut la position) ;
  2. sinon, si D > 0,2 → `span_W` (la direction lente suit les conditions : piège) ;
  3. sinon → `lenteur`.
Cas (performance = Spearman médian sur les unités de test ; C-MAPSS : sur le régime cible) :
  - N-CMAPSS DS01, DS05, DS07 : classe de vol non vue, sans dérive ; et dérive d'usage modérée fabriquée
    (vB05, h = 0,30 ; la forte est écartée : confondue avec la perte de données, vB05b) ;
  - C-MAPSS FD002, FD004 : les 6 cas de a6 (médiane des 5 tirages ; résultats de vB19 et vB21) ;
  - batteries : 4 groupes d'usage non vus (résultats de vB22 ; mêmes indicateurs pour les 4).
XJTU-SY exclu (W constant par roulement : p_hors non défini).
Mesure : regret = meilleure performance − performance de la méthode choisie. Comparaison : règle contre
« toujours lenteur », « toujours modèle », « toujours span_W ». Aussi : Spearman(p_hors, gain du modèle sur
la lenteur) sur tous les cas.

Lecture fixée AVANT de mesurer : la règle est utile si son regret médian ≤ 0,02 ET son regret maximal est
nettement inférieur (≥ 0,2 de moins) à celui de la meilleure stratégie fixe. Sinon, on décrit.
Réserve : les seuils (0,3 ; 0,2) sont fixés d'après les jeux déjà vus (batteries 0,10 ; N-CMAPSS ≥ 0,49 ;
D ≈ 0 sans dérive) : c'est une vérification de cohérence, pas une validation hors échantillon.

Machine : Mac. Sortie : voie_B_avec_exo/results/vB24_*.csv, vB24.txt
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB05_derive_usage as d5  # noqa: E402
import vB08_espace_conditions as v8  # noqa: E402
import vB14_r_sans_etiquettes as r14  # noqa: E402
import vB15_detecteur_piege as d15  # noqa: E402
import vB19_cmapss_avec_w as b19  # noqa: E402
import vB22_batteries as b22  # noqa: E402
import vB23_part_hors_sans_etiquettes as v23  # noqa: E402
import sfa  # noqa: E402
from evaluateur import evaluer  # noqa: E402

P_SEUIL, D_SEUIL = 0.3, 0.2


def regle(p, D):
    return "modele" if p < P_SEUIL else ("span_W" if D > D_SEUIL else "lenteur")


def indicateurs(S, W, unit, cycle):
    s = sfa.ajuster(S, unit, cycle).score(S)
    return dict(p_hors=v23.estimer(S, W, unit, cycle)["tendance"], R_chapeau=r14.r_chapeau(s, unit, cycle),
                D=d15.detecteur(s, W, unit, cycle))


def phi(W):
    if W.shape[1] < 2:                     # une seule variable : pas de produits croisés (vB30)
        return np.hstack([W, W ** 2])
    prod = np.stack([W[:, i] * W[:, j] for i, j in combinations(range(W.shape[1]), 2)], 1)
    return np.hstack([W, W ** 2, prod])


def ncmapss():
    rows = []
    for ds in d5.DS_:
        v.it10.DILATION = v.parse_key(ds)[1]
        fc, train, ref, unseen = v.PLAN[ds]
        df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
        complets = {u: v.fenetres(df, u) for u in [ref] + list(unseen)}
        for nom_sc in ["aucune", "montante_h0.30"]:
            tr = pd.concat([d5.fenetres_derive(df, u, d5.SCENARIOS[nom_sc])[0] for u in train], ignore_index=True)
            S, W, un, cy = tr[v.S_COLS].values, tr[v.W_COLS].values, tr.unit.values, tr.cycle.values
            ind = indicateurs(S, W, un, cy)
            muW, sdW = W.mean(0), W.std(0)
            q = LinearRegression().fit(phi((W - muW) / sdW), S)
            m_mod = sfa.ajuster(S - q.predict(phi((W - muW) / sdW)), un, cy)
            mods = {"lenteur": lambda x, m=sfa.ajuster(S, un, cy): m.score(x[v.S_COLS].values),
                    "modele": lambda x: m_mod.score(x[v.S_COLS].values
                                                    - q.predict(phi((x[v.W_COLS].values - muW) / sdW))),
                    "span_W": lambda x, m=v8.span_W(S, W, un, cy, reduire=False)[0]: m.score(x[v.S_COLS].values)}
            perf = {}
            for nom, f in mods.items():
                fen = pd.concat([x.assign(score=f(x)) for x in list(complets.values()) + [tr]], ignore_index=True)
                res = evaluer(fen, ds=ds, train_units=train, ref_unit=ref, non_vues=list(unseen))
                perf[nom] = float(res[res.role == "non_vue"].corr_spearman.median())
            rows.append(dict(jeu="N-CMAPSS", cas=f"{ds} {nom_sc}", **ind, **perf))
    return rows


def cmapss():
    rows = []
    r19 = pd.read_csv(v.OUT / "vB19_cmapss.csv")
    r21 = pd.read_csv(v.OUT / "vB21_d_cmapss.csv")
    for fd in ["FD002", "FD004"]:
        d = b19.a6.charger(fd)
        for cas, retire, cible in b19.a6.CAS:
            inds = []
            for tirage in range(b19.N_TIRAGES):
                rng = np.random.default_rng(tirage)
                u = d.unit.unique()
                tr = d[d.unit.isin(set(rng.choice(u, size=len(u) // 2, replace=False)))]
                if retire is not None:
                    tr = tr[tr.regime != retire]
                X = [c for c in b19.a6.COLS[5:] if tr[c].std() > 1e-8]
                Wk = [c for c in b19.W_C if tr[c].std() > 1e-8]
                inds.append(indicateurs(tr[X].values.astype(float), tr[Wk].values.astype(float),
                                        tr.unit.values, tr.cycle.values))
            ind = pd.DataFrame(inds).median().to_dict()
            x = r19[(r19.FD == fd) & (r19.cas == cas)]
            per = x.groupby(["tirage", "methode"]).sp_cible.median().unstack("methode").median()
            sw = r21[(r21.FD == fd) & (r21.cas == cas) & (r21.methode == "span_W v2")].sp_cible.median()
            rows.append(dict(jeu="C-MAPSS", cas=f"{fd} {cas}", **ind, lenteur=per["sfa_brut"],
                             modele=per["sfa_resid_quad"], span_W=sw))
    return rows


def batteries():
    df = b22.charger()
    tr = df[df.bat.isin(b22.GROUPES["uniforme"])]
    ind = indicateurs(tr[b22.S_B].values, tr[b22.W_B].values, tr.bat.values, tr.cycle.values)
    r = pd.read_csv(v.OUT / "vB22_batteries.csv")
    r = r[r.cadre == "classe non vue"]
    rows = []
    for g, x in r.groupby("groupe"):
        per = x.groupby("methode").sp_capacite.median()
        rows.append(dict(jeu="batteries", cas=g, **ind, lenteur=per["sfa_brut"], modele=per["sfa_resid_quad"],
                         span_W=per["span_W"]))
    return rows


def main():
    t = pd.DataFrame(ncmapss() + cmapss() + batteries())
    M = ["lenteur", "modele", "span_W"]
    t["meilleure"] = t[M].max(axis=1)
    t["choix"] = [regle(p, D) for p, D in zip(t.p_hors, t.D)]
    t["regret_regle"] = t.meilleure - t.apply(lambda r: r[r.choix], axis=1)
    for m in M:
        t[f"regret_{m}"] = t.meilleure - t[m]
    t.to_csv(v.OUT / "vB24_regle_choix.csv", index=False)
    pd.set_option("display.width", 250)
    print(t[["jeu", "cas", "p_hors", "R_chapeau", "D", "lenteur", "modele", "span_W", "choix", "regret_regle"]]
          .round(2).to_string(index=False))
    print("\n=== regret : médiane / maximum ===")
    for c in ["regret_regle"] + [f"regret_{m}" for m in M]:
        print(f"{c:18s} médiane {t[c].median():.3f}   max {t[c].max():.3f}")
    print(f"\nSpearman(p_hors, gain du modèle sur la lenteur) = "
          f"{spearmanr(t.p_hors, t.modele - t.lenteur).statistic:.2f} sur {len(t)} cas")


if __name__ == "__main__":
    main()
