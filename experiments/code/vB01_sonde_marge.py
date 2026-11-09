"""vB01 — Étape 1 : la baisse de H-CDE sur une classe non vue est-elle rattrapable ?

SONDE SUPERVISÉE = DIAGNOSTIC, JAMAIS UNE MÉTHODE (DB-01).

Question : face à une classe de vol non vue, H-CDE passe de ~0,9 à ~0,7 d'AUROC (it18, par vol). Le
signal de ces unités contient-il encore l'information de dégradation (la perte est alors due au
modèle, la voie B a une marge), ou est-il intrinsèquement moins informatif dans ces conditions
(physique : la sensibilité des capteurs dépend du point de fonctionnement ; la perte n'est alors pas
rattrapable) ?

Sonde : pour CHAQUE unité évaluée (référence + non vues) SÉPARÉMENT, on apprend la dégradation vraie à
partir des moyennes de fenêtre des 14 capteurs et des 4 variables d'opération (mêmes fenêtres que
H-CDE, it12), avec les étiquettes DE CETTE UNITÉ. Prédictions hors-pli sur tous ses vols, puis
évaluateur commun (par vol, D-24). Deux sondes, déclarées d'avance :
  - ridge (linéaire, standardisé) ;
  - gradient boosting (HistGradientBoosting, non linéaire : interactions capteur × condition).
Validation croisée par BLOCS de 10 vols contigus, répartis en 5 plis de façon cyclique ; les fenêtres
d'entraînement dont les vols d'entrée chevauchent ceux d'une fenêtre de test sont PURGÉES (sinon la
sonde verrait les mêmes données brutes).

Limites, déclarées avant mesure :
  - la sonde ne voit que des moyennes de fenêtre (H-CDE voit la série) : c'est une borne inférieure
    de l'information disponible, pas une borne supérieure stricte. Si elle réussit, la marge existe ;
    si elle échoue, c'est peu concluant sans sonde plus riche ;
  - elle interpole la dégradation entre blocs voisins de la même unité : optimiste par construction ;
  - seul T1 a un sens (chaque unité a sa propre sonde : pas de seuil commun à transférer).

Lecture fixée AVANT de mesurer (Spearman principale, D-25 ; comparée à H-CDE it18, médiane des runs
valides, même unité) :
  - sonde ≈ aussi bonne sur les non vues que sur la référence → la classe ne rend pas le signal moins
    informatif ; la perte de H-CDE vient du modèle : MARGE ;
  - sonde aussi dégradée que H-CDE sur les non vues (≤ H-CDE) → limite d'information ou de physique :
    PAS DE MARGE avec ces entrées ;
  - entre les deux → marge partielle ; on la chiffre, on ne conclut pas au-delà.

Machine : Mac, quelques minutes. Sortie : voie_B_avec_exo/results/vB01_sonde_par_vol.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VOIE = Path(__file__).resolve().parents[1]
TRONC = VOIE.parent
sys.path.insert(0, str(TRONC / "code"))
import it10_hcde_covariate as it10  # noqa: E402
from campagne_ds import PLAN, parse_key  # noqa: E402
from evaluateur import W_COLS, evaluer  # noqa: E402
from it10_hcde_covariate import FEATS, W_VARS, X_VARS, unit_windows  # noqa: E402
from it12_campagne_eval import DATASETS  # noqa: E402

OUT = VOIE / "results"
RUNS = {"DS01": "DS01", "DS05": "DS05"}     # seuls DS avec des runs H-CDE valides (it18)
BLOC, N_PLIS = 10, 5
S_COLS = [f"S_{k}" for k in X_VARS]
T1 = ["AUROC", "corr_spearman", "corr_pearson", "FPR95"]


def fenetres(df: pd.DataFrame, unit: int) -> pd.DataFrame:
    """Comme it12.window_meta (mêmes bornes), avec en plus le vol où la fenêtre COMMENCE (purge)."""
    d = df[df.unit == unit].reset_index(drop=True)
    X = d[FEATS].values.astype(np.float64)
    cyc = d["cycle"].values.astype(int)
    rows = []
    for (s, e) in unit_windows(X):
        m = X[s:e].mean(0)
        rows.append(dict(unit=unit, cycle=int(cyc[e]), cycle_debut=int(cyc[s]),
                         **{f"W_{k}": m[j] for j, k in enumerate(W_VARS)},
                         **{f"S_{k}": m[len(W_VARS) + j] for j, k in enumerate(X_VARS)}))
    return pd.DataFrame(rows)


def sondes():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gbm": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, early_stopping=False,
                                             random_state=0),
    }


def hors_pli(f: pd.DataFrame, y: np.ndarray, sonde) -> tuple[np.ndarray, float]:
    """Prédictions hors-pli, blocs de vols contigus, purge des chevauchements d'entrée."""
    X = f[S_COLS + W_COLS].values
    c0, c1 = f.cycle_debut.values, f.cycle.values
    pli = ((c1 - c1.min()) // BLOC) % N_PLIS
    pred = np.full(len(f), np.nan)
    purge = []
    for k in range(N_PLIS):
        te = pli == k
        if not te.any():
            continue
        couverts = np.zeros(c1.max() + 2, bool)
        for a, b in zip(c0[te], c1[te]):
            couverts[a:b + 1] = True
        cum = np.concatenate([[0], np.cumsum(couverts)])
        chevauche = (cum[c1 + 1] - cum[c0]) > 0
        tr = ~te & ~chevauche
        purge.append(1 - tr.sum() / (~te).sum())
        pred[te] = sonde.fit(X[tr], y[tr]).predict(X[te])
    return pred, float(np.mean(purge))


def main():
    OUT.mkdir(exist_ok=True)
    it18 = pd.read_csv(TRONC / "results/it18_references_par_vol.csv")
    lignes = []
    for ds, key in RUNS.items():
        it10.DILATION = parse_key(key)[1]
        fc, train, ref, unseen = PLAN[ds]
        df = pd.read_csv(DATASETS / f"N-CMAPSS_{ds}.csv")
        lab = df.groupby(["unit", "cycle"]).degradation.first().abs()
        f = {u: fenetres(df, u) for u in list(train) + [ref] + list(unseen)}
        for u, x in f.items():
            n_vols = x.cycle.nunique()
            span = (x.cycle - x.cycle_debut).median()
            print(f"{ds} u{u} : {len(x)} fenêtres, {n_vols} vols, une fenêtre couvre ~{span:.0f} vols")
        for nom, sonde in sondes().items():
            parts = []
            for u, x in f.items():
                x = x.copy()
                if u in train:                       # sert seulement à W_ref (fausses alarmes intra)
                    x["score"], p = 0.0, np.nan
                else:
                    y = lab.loc[list(zip(x.unit, x.cycle))].values
                    x["score"], p = hors_pli(x, y, sonde)
                    print(f"  {nom} u{u} : {100 * p:.0f} % des fenêtres d'entraînement purgées")
                parts.append(x.drop(columns="cycle_debut"))
            res = evaluer(pd.concat(parts, ignore_index=True), ds=ds, train_units=train,
                          ref_unit=ref, non_vues=list(unseen))
            lignes.append(res.assign(methode=f"sonde_{nom} (étiquettes)", DS=ds))
    t = pd.concat(lignes, ignore_index=True)
    t.to_csv(OUT / "vB01_sonde_par_vol.csv", index=False)

    # --- vue globale : unité par unité (D5), sonde contre H-CDE et naïf (médiane des runs valides) ---
    h = (it18.groupby(["methode", "DS", "unite"])[T1].median().reset_index())
    s = t[["methode", "DS", "unite", "role"] + T1]
    v = pd.concat([s, h.merge(s[["DS", "unite", "role"]].drop_duplicates(), on=["DS", "unite"])])
    v["classe"] = v.apply(lambda r: "Fc1" if r.role == "ref" else f"Fc{PLAN[r.DS][3][r.unite]}", axis=1)
    pd.set_option("display.width", 200)
    for m in ["corr_spearman", "AUROC", "corr_pearson"]:
        print(f"\n=== {m}, par vol ===")
        print(v.pivot_table(index=["DS", "unite", "classe"], columns="methode", values=m)
              .round(3).to_string())
    print(f"\nécrit : {OUT / 'vB01_sonde_par_vol.csv'}")


if __name__ == "__main__":
    main()
