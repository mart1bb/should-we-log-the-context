"""vB12 — Diagramme de phase : QUAND les variables d'opération servent-elles ?

vB11 : sur le banc calibré sur N-CMAPSS (rapport signal de dégradation / bruit par vol R_deg ≈ 21), la
SFA sans W n'est pas piégée par la dérive d'usage, même forte ; sur un banc plus bruité (vB09), elle
l'est et `span_W` tient. Hypothèse : W sert quand la dérive est plus lente que la dégradation
RELATIVEMENT au bruit par vol, donc à faible R_deg et forte dérive.

Plan, déclaré avant mesure : banc calibré de vB11 (mêmes nuisances ; λ = 1 ; σ_f = 0,25), deux axes :
  - rapport signal/bruit : se ∈ {0,0075 ; 0,02 ; 0,043 ; 0,09 ; 0,16}, visant R_deg ≈ 145 / 21 / 4,3 / 1 /
    0,3 (DS01 / DS07 / DS05 / plus bruité) ; R_deg réel mesuré sur le banc sans dérive (même mesure
    que vB11) ;
  - dérive : δ ∈ {0, 1, 3, 10}, soit un rapport variance de dérive / variance rapide κ = δ²/(12 σ_f²) ≈
    0 / 1,3 / 12 / 133.
Méthodes : `sfa_14` (SFA sans W, `sfa.ajuster` : la méthode de A), `span_W` (vB08, orientation à conditions
égales), sonde supervisée (plafond, DB-01). 10 graines ; Spearman médian des 2 unités non vues.
Phases (règle fixée d'avance, sur les médianes) :
  - « lenteur suffit » : `sfa_14` ≥ 0,8 ;
  - « W sert » : `sfa_14` < 0,8 et `span_W` ≥ `sfa_14` + 0,2 ;
  - « W ne suffit pas » : `span_W` < 0,5 et plafond ≥ 0,7 ;
  - « pas d'information » : plafond < 0,7 ;
  - sinon « intermédiaire ».

Lecture fixée AVANT de mesurer : une zone « W sert » existe (faible R_deg, forte dérive) ; on situe sa
frontière en R_deg et on y place DS01 (145), DS07 (21), DS05 (4,3) — mesurés en vB11, sans dérive
naturelle (a1 de A). Si aucune zone « W sert » n'existe, la contribution méthode de B tombe ; on le dit.

Machine : Mac. Sortie : voie_B_avec_exo/results/vB12_*.csv, vB12.txt
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
import vB01_sonde_marge as v  # noqa: E402
import vB08_espace_conditions as v8  # noqa: E402
import vB10_banc_realiste as b10  # noqa: E402
import vB11_calibration as c11  # noqa: E402
import sfa  # noqa: E402

SIGMA_F = 0.25
SES = [0.0075, 0.02, 0.043, 0.09, 0.16]
DELTAS = [0.0, 1.0, 3.0, 10.0]
CAL = dict(sW=0.1, sh=0.5, sm=0.0107, het=0.29, se=0.02, so=0.0, a=1.0)     # vB11


def phase(r):
    if r.plafond < 0.7:
        return "pas d'information"
    if r.sfa_14 >= 0.8:
        return "lenteur suffit"
    if r.span_W >= r.sfa_14 + 0.2:
        return "W sert"
    if r.span_W < 0.5:
        return "W ne suffit pas"
    return "intermédiaire"


def main():
    rows, rdeg = [], []
    for se in SES:
        nz = dict(CAL); nz["se"] = se
        m = []
        for g in range(b10.N_GRAINES):
            df = b10.generer(g, 0.0, SIGMA_F, nz)
            tr = df[df.role == "train"]
            m.append(c11.mesures(tr[b10.S_COLS].values, tr[b10.W_COLS].values, tr.degradation.values,
                                 tr.unit.values)["R_deg"])
        rdeg.append(dict(se=se, R_deg=float(np.median(m))))
        for delta, g in product(DELTAS, range(b10.N_GRAINES)):
            df = b10.generer(g, delta, SIGMA_F, nz)
            tr = df[df.role == "train"]
            X, W, un, cy = tr[b10.S_COLS].values, tr[b10.W_COLS].values, tr.unit.values, tr.cycle.values
            ms = v8.span_W(X, W, un, cy)[0]
            mods = {"sfa_14": sfa.ajuster(X, un, cy),
                    "span_W": sfa.SFA(ms.moyenne, ms.ecart_type, b10.v8b.signe_partiel(ms, tr) * ms.w)}
            F = b10.S_COLS + b10.W_COLS
            sonde = Ridge(alpha=1.0).fit(tr[F].values, tr.degradation.values)
            for u in (5, 6):
                x = df[df.unit == u]
                rows.append(dict(se=se, delta=delta, graine=g, methode="plafond",
                                 spearman=spearmanr(sonde.predict(x[F].values), x.degradation).statistic))
                for nom, mm in mods.items():
                    rows.append(dict(se=se, delta=delta, graine=g, methode=nom,
                                     spearman=spearmanr(mm.score(x[b10.S_COLS].values), x.degradation).statistic))
    t = pd.DataFrame(rows)
    rd = pd.DataFrame(rdeg)
    t.to_csv(v.OUT / "vB12_diagramme_phase.csv", index=False)
    rd.to_csv(v.OUT / "vB12_rdeg.csv", index=False)
    med = t.groupby(["se", "delta", "methode"]).spearman.median().unstack("methode").reset_index()
    med = med.merge(rd, on="se")
    med["kappa"] = med.delta ** 2 / (12 * SIGMA_F ** 2)
    med["phase"] = med.apply(phase, axis=1)
    pd.set_option("display.width", 220)
    print("=== R_deg mesuré sur le banc, par bruit capteurs (sans dérive) ===")
    print(rd.round(2).to_string(index=False))
    print("\n=== médianes (non vues) et phase ===")
    print(med[["R_deg", "kappa", "sfa_14", "span_W", "plafond", "phase"]].round(2).to_string(index=False))
    print("\n=== carte des phases (lignes : R_deg ; colonnes : κ dérive / variation rapide) ===")
    carte = med.pivot_table(index="R_deg", columns="kappa", values="phase", aggfunc="first")
    print(carte.sort_index(ascending=False).to_string())
    print("\nPlacement N-CMAPSS (vB11, sans dérive naturelle) : DS01 R_deg 145, DS07 21, DS05 4,3.")


if __name__ == "__main__":
    main()
