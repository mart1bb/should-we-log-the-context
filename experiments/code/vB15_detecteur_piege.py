"""vB15 — Détecter, sans étiquettes et grâce à W, que la SFA sans W est piégée par une dérive d'usage.

Contexte : R̂ (vB14) dit, sans étiquettes, si le signal le long de la direction de la SFA domine le
bruit ; mais si la SFA est piégée par une dérive d'usage, sa direction suit la dérive et R̂ est élevé à
tort. Il faut un second indicateur, qui dise si la direction trouvée suit les CONDITIONS.

Détecteur, déclaré avant mesure : sur les unités d'entraînement, par vol, score s et caractéristiques
φ(W) (W, W², produits, standardisées) ; différences entre vols consécutifs d'une même unité ;
    D = R²( Δs ~ Δφ(W) )   (moindres carrés, sans constante).
Pourquoi les différences : sous dérive, la tendance LENTE de W est corrélée à l'âge, donc aussi à la
dégradation — un bon score la suivrait aussi. La variation RAPIDE de W, elle, n'a rien à voir avec la
dégradation : si le score y réagit, c'est qu'il suit les conditions. Aucune étiquette.

Où :
  - banc de vB12 (σ_f = 0,25 ; bruit capteurs × dérive ; 10 graines) : SFA sans W (`sfa.ajuster`) ;
    vérité = Spearman(score, d) sur les non vues ; « piégée » = Spearman < 0,5 ;
  - N-CMAPSS, DS01 / DS05 / DS07 : SFA sans W sur les scénarios de dérive fabriquée de vB05 (aucune,
    montante h = 0,30 / 0,15, descendante h = 0,15), unités d'entraînement dérivées, non vues complètes.

Lecture fixée AVANT de mesurer :
  - banc : D sépare piégées et non piégées (AUROC ≥ 0,9) → D est un détecteur de piège sans étiquettes ;
  - N-CMAPSS : D bas sans dérive, haut là où la SFA échoue sous dérive (vB05) ;
  - on rapporte aussi R̂ (vB14) pour montrer qu'il ne voit PAS le piège.

Machine : Mac. Sortie : voie_B_avec_exo/results/vB15_*.csv, vB15.txt
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB05_derive_usage as d5  # noqa: E402
import vB08_espace_conditions as v8  # noqa: E402
import vB10_banc_realiste as b10  # noqa: E402
import vB12_diagramme_phase as p12  # noqa: E402
import vB14_r_sans_etiquettes as r14  # noqa: E402
import sfa  # noqa: E402
from evaluateur import auroc  # noqa: E402


def detecteur(s: np.ndarray, W: np.ndarray, unit, cycle) -> float:
    Wz = (W - W.mean(0)) / W.std(0)
    F = v8.phi(Wz)
    F = (F - F.mean(0)) / (F.std(0) + 1e-12)
    fv = sfa.moyennes_par_vol(np.column_stack([s, F]), unit, cycle).sort_values(["unit", "cycle"])
    ds_, dF = [], []
    for _, g in fv.groupby("unit"):
        ds_.append(np.diff(g[0].values))
        dF.append(np.diff(g[list(range(1, F.shape[1] + 1))].values, axis=0))
    ds_, dF = np.concatenate(ds_), np.vstack(dF)
    coef, *_ = np.linalg.lstsq(dF, ds_, rcond=None)
    return float(1 - np.var(ds_ - dF @ coef) / np.var(ds_))


def main():
    rows = []
    for se, delta, g in product(p12.SES, p12.DELTAS, range(b10.N_GRAINES)):
        nz = dict(p12.CAL); nz["se"] = se
        df = b10.generer(g, delta, p12.SIGMA_F, nz)
        tr = df[df.role == "train"]
        X, W, un, cy = tr[b10.S_COLS].values, tr[b10.W_COLS].values, tr.unit.values, tr.cycle.values
        m = sfa.ajuster(X, un, cy)
        s = m.score(X)
        verite = np.median([spearmanr(m.score(df[df.unit == u][b10.S_COLS].values),
                                      df[df.unit == u].degradation).statistic for u in (5, 6)])
        rows.append(dict(lieu="banc", se=se, delta=delta, graine=g, D=detecteur(s, W, un, cy),
                         R_chapeau=r14.r_chapeau(s, un, cy), spearman_nv=verite))
    for ds in d5.DS_:
        v.it10.DILATION = v.parse_key(ds)[1]
        fc, train, ref, unseen = v.PLAN[ds]
        df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
        complets = {u: v.fenetres(df, u) for u in unseen}
        lab = df.groupby(["unit", "cycle"]).degradation.first().abs()
        for nom_sc, derive in d5.SCENARIOS.items():
            tr = pd.concat([d5.fenetres_derive(df, u, derive)[0] for u in train], ignore_index=True)
            X, W, un, cy = tr[v.S_COLS].values, tr[v.W_COLS].values, tr.unit.values, tr.cycle.values
            m = sfa.ajuster(X, un, cy)
            s = m.score(X)
            ver = []
            for u, x in complets.items():
                e = sfa.moyennes_par_vol(m.score(x[v.S_COLS].values)[:, None], x.unit.values, x.cycle.values)
                ver.append(spearmanr(e[0], lab.loc[list(zip(e.unit, e.cycle))].values).statistic)
            rows.append(dict(lieu=ds, scenario=nom_sc, D=detecteur(s, W, un, cy),
                             R_chapeau=r14.r_chapeau(s, un, cy), spearman_nv=float(np.median(ver))))
    t = pd.DataFrame(rows)
    t.to_csv(v.OUT / "vB15_detecteur_piege.csv", index=False)
    pd.set_option("display.width", 220)
    banc = t[t.lieu == "banc"].copy()
    banc["piegee"] = banc.spearman_nv < 0.5
    print("=== banc : médianes par cellule ===")
    print(banc.groupby(["se", "delta"])[["D", "R_chapeau", "spearman_nv"]].median().round(2).to_string())
    pos, neg = banc[banc.piegee], banc[~banc.piegee]
    print(f"\nbanc : {len(pos)} piégées / {len(neg)} non piégées ; "
          f"AUROC(D) = {auroc(pos.D.values, neg.D.values):.2f} ; "
          f"AUROC(1/R̂) = {auroc(1 / pos.R_chapeau.values, 1 / neg.R_chapeau.values):.2f}")
    print("\n=== N-CMAPSS (SFA sans W, scénarios de dérive de vB05) ===")
    print(t[t.lieu != "banc"][["lieu", "scenario", "D", "R_chapeau", "spearman_nv"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
