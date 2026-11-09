"""vB14 — Estimer sans étiquettes le rapport signal/bruit qui décide si W sert (carte de vB12).

vB12/vB12b : W n'aide la SFA que si le rapport signal de dégradation / bruit hors de l'espace des
conditions (R⊥) est faible et la dérive d'usage extrême. R⊥ se mesure avec les étiquettes (DB-01) :
inutilisable sur une flotte réelle.

Estimateur sans étiquettes, déclaré avant mesure : sur les unités d'entraînement, score SFA par vol s
(SFA commune, capteurs seuls, moyennes par vol). Si s = signal lent + bruit indépendant d'un vol à
l'autre, les différences entre vols consécutifs ne contiennent presque que le bruit :
    σ̂² = Var_intra(Δs) / 2 ,   R̂ = (Var_intra(s) − σ̂²) / σ̂²
(intra = après retrait de la moyenne de chaque unité). Équivalent : R̂ = 2/λ − 1, λ = lenteur de la
direction trouvée. R̂ mesure le signal/bruit LE LONG de la direction de la SFA ; R⊥ le mesure sur tout le
complément de l'espace des conditions — les deux diffèrent par la normalisation ; on attend une relation
monotone, pas l'égalité.
Deux lieux :
  - N-CMAPSS, DS01 / DS05 / DS07 (et DS04 pour info), unités d'entraînement, SFA sur les fenêtres de la
    campagne v3 (comme vB04) ;
  - banc de vB12 (σ_f = 0,25, sans dérive), pour chaque niveau de bruit capteurs, 10 graines.
Comparaison avec R⊥ (vB12b) et avec les phases de vB12.

Lecture fixée AVANT de mesurer :
  - R̂ croît avec R⊥ (Spearman ≥ 0,9 sur les 5 niveaux du banc + 3 DS) → R̂ peut remplacer R⊥ comme axe
    de la carte, sans étiquettes ;
  - on relit la carte de vB12 en R̂ et on y replace DS01/05/07 ;
  - sinon : l'estimateur ne suit pas R⊥, on décrit.

Machine : Mac. Sortie : voie_B_avec_exo/results/vB14_*.csv, vB14.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB02b_support_et_modes as b  # noqa: E402
import vB10_banc_realiste as b10  # noqa: E402
import vB12_diagramme_phase as p12  # noqa: E402
import sfa  # noqa: E402


def r_chapeau(s: np.ndarray, unit: np.ndarray, cycle: np.ndarray) -> float:
    fv = sfa.moyennes_par_vol(s[:, None], unit, cycle).sort_values(["unit", "cycle"])
    var_s, var_d = [], []
    for _, g in fv.groupby("unit"):
        x = g[0].values
        var_s.append(np.var(x - x.mean()))
        var_d.append(np.var(np.diff(x)))
    sig2 = np.mean(var_d) / 2
    return float((np.mean(var_s) - sig2) / sig2)


def main():
    rows = []
    for ds in b.CLES:
        d = b.charger(ds)
        tr = pd.concat([d["f"][u] for u in d["train"]], ignore_index=True)
        X, un, cy = tr[v.S_COLS].values, tr.unit.values, tr.cycle.values
        m = sfa.ajuster(X, un, cy)
        rows.append(dict(source=f"{ds} (fenêtres)", R_chapeau=r_chapeau(m.score(X), un, cy)))
        # Correction (ajoutée après la 1re exécution, déclarée) : les fenêtres couvrent ~5 vols et se
        # chevauchent ; les vols consécutifs y sont corrélés, ce qui sous-estime le bruit. Même entrée que
        # R⊥ (vB12b) : moyennes BRUTES par vol (toutes les lignes du vol).
        fc, train, ref, unseen = v.PLAN[ds]
        df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
        fv = df[df.unit.isin(train)].groupby(["unit", "cycle"])[[c[2:] for c in v.S_COLS]].mean().reset_index()
        Xf = fv[[c[2:] for c in v.S_COLS]].values.astype(np.float64)
        mf = sfa.ajuster(Xf, fv.unit.values, fv.cycle.values)
        rows.append(dict(source=ds, R_chapeau=r_chapeau(mf.score(Xf), fv.unit.values, fv.cycle.values)))
    for se in p12.SES:
        nz = dict(p12.CAL); nz["se"] = se
        rr = []
        for g in range(b10.N_GRAINES):
            df = b10.generer(g, 0.0, p12.SIGMA_F, nz)
            tr = df[df.role == "train"]
            X, un, cy = tr[b10.S_COLS].values, tr.unit.values, tr.cycle.values
            rr.append(r_chapeau(sfa.ajuster(X, un, cy).score(X), un, cy))
        rows.append(dict(source=f"banc se={se}", R_chapeau=float(np.median(rr))))
    t = pd.DataFrame(rows)
    # R⊥ de vB12b (étiquettes)
    rperp = {"DS01": 283.74, "DS05": 2.88, "DS07": 57.94, "banc se=0.0075": 7.27, "banc se=0.02": 5.33,
             "banc se=0.043": 2.09, "banc se=0.09": 0.68, "banc se=0.16": 0.23}
    t["R_perp (étiquettes)"] = t.source.map(rperp)
    t.to_csv(v.OUT / "vB14_r_sans_etiquettes.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(2).to_string(index=False))
    c = t.dropna()
    print(f"\nSpearman(R̂, R⊥) sur {len(c)} points : {spearmanr(c.R_chapeau, c['R_perp (étiquettes)']).statistic:.2f}")


if __name__ == "__main__":
    main()
