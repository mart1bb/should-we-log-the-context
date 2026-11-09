"""a6 — C-MAPSS FD002/FD004 : un régime d'opération retiré de l'entraînement (plan gelé : JOURNAL, 24 sept. ;
CANAL M-19).

Question : la SFA sans W, entraînée sans un régime, classe-t-elle la dégradation sur ce régime (prop. 2),
et son niveau sain y est-il décalé (corollaire : le niveau casse avant le classement) ?
W ne sert qu'à étiqueter les régimes (découpage, évaluation), jamais en entrée.

Cas : C0_* (tous les régimes vus, comme Xue et al. 2024 ; même régime cible que R*), R42 (42 kft retiré, hors support en altitude),
R60 (manette 60 retirée, hors support en manette), R20 (20 kft retiré, interpolation).
Sortie : results/a6_cmapss.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parents[1] / "code"))
import sfa  # noqa: E402

import os
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ICI.parent / "data"))  # voir README
DONNEES = DATA_ROOT / "CMAPSS/archive"
OUT = ICI.parent / "results"
COLS = ["unit", "cycle", "s1", "s2", "s3"] + [f"x{i}" for i in range(1, 22)]
R42, R60, R20 = "42_0.84_100", "25_0.62_60", "20_0.7_100"
# (nom, régime retiré de l'entraînement, régime cible mesuré) ; contrôles C0 : tout vu, même cible
CAS = [("C0_42", None, R42), ("C0_60", None, R60), ("C0_20", None, R20),
       ("R42", R42, R42), ("R60", R60, R60), ("R20", R20, R20)]
N_TIRAGES, DEBUT = 5, 0.2


def charger(fd):
    d = pd.read_csv(DONNEES / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    d["regime"] = (d.s1.round(0).astype(int).astype(str) + "_" + d.s2.round(2).astype(str) + "_"
                   + d.s3.round(0).astype(int).astype(str))
    d["frac"] = d.cycle / d.groupby("unit").cycle.transform("max")
    return d


def v_c(Z, d):
    """ACP 99 % des cycles de début de vie, centrés par unité (a4/P1)."""
    m = (d.frac < DEBUT).values
    H = pd.DataFrame(Z[m]).assign(unit=d.unit.values[m])
    cols = list(range(Z.shape[1]))
    Hc = H[cols].values - H.groupby("unit")[cols].transform("mean").values
    val, vec = np.linalg.eigh(np.cov(Hc, rowvar=False))
    val, vec = val[::-1], vec[:, ::-1]
    r = int(np.searchsorted(np.cumsum(val) / val.sum(), 0.99) + 1)
    return vec[:, :r]


def evaluer_cas(d, fd, cas, retire, cible, tirage):
    rng = np.random.default_rng(tirage)
    unites = d.unit.unique()
    tr_u = set(rng.choice(unites, size=len(unites) // 2, replace=False))
    tr = d[d.unit.isin(tr_u)]
    if retire is not None:
        tr = tr[tr.regime != retire]
    te = d[~d.unit.isin(tr_u)]
    X = [c for c in COLS[5:] if tr[c].std() > 1e-8]
    m = sfa.ajuster(tr[X].values.astype(np.float64), tr.unit.values, tr.cycle.values)
    s = m.score(te[X].values.astype(np.float64))
    # naïf : Mahalanobis sur le début de vie des cycles d'entraînement
    Ztr = (tr[X].values - m.moyenne) / m.ecart_type
    Zte = (te[X].values - m.moyenne) / m.ecart_type
    deb = (tr.frac < DEBUT).values
    mu = Ztr[deb].mean(0)
    P = np.linalg.pinv(np.cov(Ztr[deb], rowvar=False))
    naif = np.sqrt(np.einsum("ij,jk,ik->i", Zte - mu, P, Zte - mu))
    # fuite hors V_c du décalage du régime cible (début de vie, unités de test)
    V = v_c(Ztr, tr)
    t = te.assign(sfa=s, naif=naif, _i=np.arange(len(te)))
    lignes = []
    cib = cible
    tdeb = t[t.frac < DEBUT]
    dec = Zte[tdeb[tdeb.regime == cib]._i.values].mean(0) - Zte[tdeb[tdeb.regime != cib]._i.values].mean(0)
    hors = dec - V @ (V.T @ dec)
    geo = dict(r=V.shape[1], fuite_part=float(np.sum(hors ** 2) / np.sum(dec ** 2)),
               fuite_norme=float(np.linalg.norm(hors)), decalage_norme=float(np.linalg.norm(dec)))
    for meth in ["sfa", "naif"]:
        for u, g in t.groupby("unit"):
            gc, gv = g[g.regime == cib], g[g.regime != cib]
            gvd, gcd = gv[gv.frac < DEBUT], gc[gc.frac < DEBUT]
            if len(gc) < 10 or len(gcd) < 3 or len(gvd) < 10:
                continue
            bruit = gvd[meth].std()
            lignes.append(dict(
                FD=fd, cas=cas, tirage=tirage, methode=meth, unite=u, **geo,
                sp_cible=spearmanr(gc.cycle, gc[meth]).statistic,
                sp_vus=spearmanr(gv.cycle, gv[meth]).statistic,
                decalage_z=(gcd[meth].median() - gvd[meth].median()) / bruit,
                FA=float((gcd[meth] > np.quantile(gvd[meth], 0.95)).mean())))
    return lignes


def main():
    lignes = []
    for fd in ["FD002", "FD004"]:
        d = charger(fd)
        for cas, retire, cible in CAS:
            for tirage in range(N_TIRAGES):
                lignes += evaluer_cas(d, fd, cas, retire, cible, tirage)
        print(fd, "fait", flush=True)
    pd.DataFrame(lignes).to_csv(OUT / "a6_cmapss.csv", index=False)
    print("écrit : a6_cmapss.csv")


if __name__ == "__main__":
    main()
