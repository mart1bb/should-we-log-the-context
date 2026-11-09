"""vB11 — Calibrer le banc synthétique sur N-CMAPSS, puis refaire le banc réaliste.

vB10 : les niveaux de nuisance choisis à la main rendaient le banc non informatif (vB10b : même une
sonde supervisée n'y lisait plus la dégradation). On mesure donc sur N-CMAPSS les grandeurs
correspondantes, et on règle le banc pour qu'il ait les mêmes.

Grandeurs mesurées, déclarées avant mesure (sans unité, calculées de la MÊME façon sur N-CMAPSS et sur
le banc ; diagnostic avec étiquettes, DB-01) — sur les vols des unités d'entraînement, moyennes par vol
des capteurs standardisés, régression par moindres carrés
    S̄ ~ constante par unité + φ(W̄) (W, W², produits) + d
qui sépare une part « conditions » Pc, une part « dégradation » Pd et un reste E :
  - R_deg = Var(Pd) / Var(E)   (signal de dégradation / bruit par vol, sommé sur les capteurs) ;
  - R_cond = Var(Pc) / Var(Pd) (domination des conditions sur la dégradation) ;
  - sm = écart-type de (d − médiane glissante de d sur 5 vols) / amplitude de d sur la vie (bruit de
    procédé, par unité) ;
  - het : angle moyen entre les signatures de dégradation estimées unité par unité (régression par
    unité) → het = tan(angle) dans le générateur.
Sur N-CMAPSS : DS01, DS05, DS07, moyennes BRUTES par vol (toutes les lignes du vol), comme le banc
(un vecteur par vol). On retient la médiane des trois DS.

Calage du banc (σ_f = 1, λ = 1, sans dérive, sans autre nuisance) : recherche sur une grille de
(se, a) — bruit capteurs, amplitude de la signature — du couple dont (R_deg, R_cond), médianes sur
10 graines, est le plus proche de N-CMAPSS (distance sur les logarithmes).
Nuisances non mesurables sur N-CMAPSS, gardées au niveau « réaliste » de vB10 et déclarées comme
telles : bruit sur W (0,1), condition non mesurée (0,5). Constantes d'unité : 0 (vB10, ablation).

Banc calibré : même plan principal que vB10 (δ ∈ {0, 3, 10} × σ_f ∈ {1 ; 0,25}, λ = 1), mêmes méthodes,
plus la sonde supervisée (plafond, DB-01).

Lecture fixée AVANT de mesurer :
  - le banc calibré est informatif : plafond (sonde) ≥ 0,7 sans dérive ; sinon N-CMAPSS lui-même
    serait très bruité par vol, et on le dit ;
  - `span_W` ≥ `sfa_14` là où la dérive domine (σ_f = 0,25, δ ≥ 3), et à ≤ 0,1 de la sonde sans dérive ;
  - sinon on décrit.

Machine : Mac. Sortie : voie_B_avec_exo/results/vB11_*.csv, vB11.txt
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402  (ajoute aussi ../code au chemin)
import vB08_espace_conditions as v8  # noqa: E402
import vB10_banc_realiste as b10  # noqa: E402

DS_ = ["DS01", "DS05", "DS07"]
SE_GRILLE = [0.005, 0.01, 0.02, 0.05, 0.1]
A_GRILLE = [0.1, 0.2, 0.5, 1.0, 2.0]


def mesures(S: np.ndarray, W: np.ndarray, d: np.ndarray, unit: np.ndarray) -> dict:
    """R_deg, R_cond, het sur des moyennes par vol (S déjà par vol)."""
    Sz = (S - S.mean(0)) / S.std(0)
    Wz = (W - W.mean(0)) / W.std(0)
    F = v8.phi(Wz)
    U = pd.get_dummies(unit).values.astype(float)
    X = np.column_stack([U, F, d])
    coef, *_ = np.linalg.lstsq(X, Sz, rcond=None)
    nU, nF = U.shape[1], F.shape[1]
    Pc = F @ coef[nU:nU + nF]
    Pd = np.outer(d, coef[-1])
    E = Sz - X @ coef
    angles = []
    sig = {}
    for u in np.unique(unit):
        m = unit == u
        Xu = np.column_stack([np.ones(m.sum()), F[m], d[m]])
        cu, *_ = np.linalg.lstsq(Xu, Sz[m], rcond=None)
        sig[u] = cu[-1] / np.linalg.norm(cu[-1])
    us = list(sig)
    for i in range(len(us)):
        for j in range(i + 1, len(us)):
            angles.append(np.degrees(np.arccos(np.clip(abs(sig[us[i]] @ sig[us[j]]), 0, 1))))
    return dict(R_deg=float(Pd.var(0).sum() / E.var(0).sum()),
                R_cond=float(Pc.var(0).sum() / Pd.var(0).sum()),
                het_angle=float(np.mean(angles)))


def bruit_procede(d: np.ndarray, unit: np.ndarray) -> float:
    r = []
    for u in np.unique(unit):
        x = pd.Series(d[unit == u])
        r.append(float((x - x.rolling(5, center=True, min_periods=1).median()).std() / (x.max() - x.min())))
    return float(np.median(r))


def mesurer_ncmapss() -> pd.DataFrame:
    rows = []
    for ds in DS_:
        fc, train, ref, unseen = v.PLAN[ds]
        df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
        df = df[df.unit.isin(train)]
        fv = df.groupby(["unit", "cycle"])[[c[2:] for c in v.S_COLS] + [c[2:] for c in v.W_COLS]
                                          + ["degradation"]].mean().reset_index()
        S = fv[[c[2:] for c in v.S_COLS]].values.astype(np.float64)
        W = fv[[c[2:] for c in v.W_COLS]].values.astype(np.float64)
        d = fv.degradation.abs().values
        r = mesures(S, W, d, fv.unit.values)
        r.update(DS=ds, sm=bruit_procede(d, fv.unit.values))
        rows.append(r)
    return pd.DataFrame(rows)


def mesurer_banc(se: float, a: float) -> dict:
    rs = []
    for g in range(b10.N_GRAINES):
        nz = dict(b10.IDEAL); nz.update(se=se, a=a)
        df = b10.generer(g, 0.0, 1.0, nz)
        tr = df[df.role == "train"]
        rs.append(mesures(tr[b10.S_COLS].values, tr[b10.W_COLS].values, tr.degradation.values,
                          tr.unit.values))
    return {k: float(np.median([r[k] for r in rs])) for k in ("R_deg", "R_cond")}


def main():
    nc = mesurer_ncmapss()
    cible = nc[["R_deg", "R_cond", "sm", "het_angle"]].median()
    print("=== N-CMAPSS (unités d'entraînement, moyennes brutes par vol) ===")
    print(nc.round(3).to_string(index=False))
    print(f"cible (médiane des 3 DS) : {cible.round(3).to_dict()}")

    grille = []
    for se, a in product(SE_GRILLE, A_GRILLE):
        m = mesurer_banc(se, a)
        dist = float(np.hypot(np.log(m["R_deg"] / cible.R_deg), np.log(m["R_cond"] / cible.R_cond)))
        grille.append(dict(se=se, a=a, **m, distance=dist))
    gr = pd.DataFrame(grille).sort_values("distance")
    print("\n=== calage du banc (σ_f = 1, λ = 1, sans dérive ni autre nuisance) : 5 meilleurs couples ===")
    print(gr.head(5).round(3).to_string(index=False))
    best = gr.iloc[0]
    het = float(np.tan(np.radians(cible.het_angle)))
    CAL = dict(b10.REALISTE)
    CAL.update(se=float(best.se), a=float(best.a), sm=float(cible.sm * 1.3), het=het)   # amplitude de d ≈ 1,3
    print(f"\nbanc calibré : {CAL}")

    rows = []
    for delta, sigma_f in product([0.0, 3.0, 10.0], [1.0, 0.25]):
        for g in range(b10.N_GRAINES):
            df = b10.generer(g, delta, sigma_f, CAL)
            tr = df[df.role == "train"]
            X, W, un, cy = tr[b10.S_COLS].values, tr[b10.W_COLS].values, tr.unit.values, tr.cycle.values
            modeles = {"sfa_14": b10.sfa.ajuster(X, un, cy),
                       "sans_W_v99": b10.sfa.ajuster_reduite(X, un, cy)[0]}
            for nom, m in [("insens_W_v99", b10.d5c.insens_v99(X, W, un, cy)),
                           ("span_W", v8.span_W(X, W, un, cy)[0])]:
                modeles[nom] = b10.sfa.SFA(m.moyenne, m.ecart_type, b10.v8b.signe_partiel(m, tr) * m.w)
            F = b10.S_COLS + b10.W_COLS
            sonde = Ridge(alpha=1.0).fit(tr[F].values, tr.degradation.values)
            for u in (5, 6):
                x = df[df.unit == u]
                rows.append(dict(delta=delta, sigma_f=sigma_f, methode="sonde (étiquettes)",
                                 spearman=spearmanr(sonde.predict(x[F].values), x.degradation).statistic))
                for nom, m in modeles.items():
                    rows.append(dict(delta=delta, sigma_f=sigma_f, methode=nom,
                                     spearman=spearmanr(m.score(x[b10.S_COLS].values), x.degradation).statistic))
    t = pd.DataFrame(rows)
    t.to_csv(v.OUT / "vB11_banc_calibre.csv", index=False)
    nc.to_csv(v.OUT / "vB11_mesures_ncmapss.csv", index=False)
    gr.to_csv(v.OUT / "vB11_calage.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== banc CALIBRÉ — Spearman médian des non vues [quartile bas] ===")
    print(t.groupby(["sigma_f", "delta", "methode"]).spearman
          .agg(lambda s: f"{s.median():.2f} [{s.quantile(0.25):.2f}]").unstack("methode")
          [["sonde (étiquettes)", "sfa_14", "sans_W_v99", "insens_W_v99", "span_W"]].to_string())


if __name__ == "__main__":
    main()
