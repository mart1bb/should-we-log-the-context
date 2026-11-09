"""vB22b — Diagnostic ajouté après vB22 (déclaré ; DB-01, avec étiquettes) : pourquoi, sur les batteries,
seul le modèle de condition marche, et la SFA sans W et `span_W` échouent ?

Hypothèse : la signature de la dégradation (perte de capacité) est surtout DANS l'espace engendré par les
effets des conditions (courant, charge débitée) — une batterie usée ressemble à une batterie plus
déchargée. Mesure, sur les batteries d'entraînement (uniformes), moyennes par cycle comme partout :
régression S ~ φ(W) + capacité ; espace des conditions V = ACP 99 % de la part φ(W) ; part de la
signature (coefficient de la capacité) HORS de V. Comparaison : N-CMAPSS (même mesure, unités
d'entraînement, moyennes brutes par vol ; la voie A mesure 47–92 % hors de son V_c en a4).
Sortie : results/vB22b.txt
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB08_espace_conditions as v8  # noqa: E402
import vB22_batteries as b22  # noqa: E402


def part_hors(S, W, d, unit):
    Sz = (S - S.mean(0)) / S.std(0)
    Wz = (W - W.mean(0)) / W.std(0)
    F = v8.phi(Wz)
    U = pd.get_dummies(unit).values.astype(float)
    X = np.column_stack([U, F, d])
    coef, *_ = np.linalg.lstsq(X, Sz, rcond=None)
    nU, nF = U.shape[1], F.shape[1]
    Pc = F @ coef[nU:nU + nF]
    a = coef[-1]
    val, vec = np.linalg.eigh(np.cov(Pc - Pc.mean(0), rowvar=False))
    val, vec = val[::-1], vec[:, ::-1]
    r = int(np.searchsorted(np.cumsum(val) / val.sum(), 0.99) + 1)
    V = vec[:, :r]
    hors = a - V @ (V.T @ a)
    return float(hors @ hors / (a @ a)), r, Sz.shape[1]


def main():
    rows = []
    df = b22.charger()
    tr = df[df.bat.isin(b22.GROUPES["uniforme"])]
    fv = tr.groupby(["bat", "cycle"])[b22.S_B + b22.W_B + ["capacite"]].mean().reset_index()
    p, r, k = part_hors(fv[b22.S_B].values, fv[b22.W_B].values, -fv.capacite.values, fv.bat.values)
    rows.append(dict(jeu="batteries (uniformes, par cycle)", part_signature_hors_V=p, r=r, capteurs=k))
    # même mesure par ÉTAPE (sans moyenne par cycle : W varie d'une étape à l'autre)
    p2, r2, _ = part_hors(tr[b22.S_B].values, tr[b22.W_B].values, -tr.capacite.values, tr.bat.values)
    rows.append(dict(jeu="batteries (uniformes, par étape)", part_signature_hors_V=p2, r=r2, capteurs=k))
    for ds in ["DS01", "DS05", "DS07"]:
        fc, train, ref, unseen = v.PLAN[ds]
        d = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
        d = d[d.unit.isin(train)]
        cols = [c[2:] for c in v.S_COLS] + [c[2:] for c in v.W_COLS] + ["degradation"]
        f = d.groupby(["unit", "cycle"])[cols].mean().reset_index()
        p, r, k = part_hors(f[[c[2:] for c in v.S_COLS]].values.astype(float),
                            f[[c[2:] for c in v.W_COLS]].values.astype(float), f.degradation.abs().values, f.unit.values)
        rows.append(dict(jeu=f"N-CMAPSS {ds} (par vol)", part_signature_hors_V=p, r=r, capteurs=k))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
