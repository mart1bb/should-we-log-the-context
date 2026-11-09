"""vB02 — Une MÊME lecture de la dégradation, apprise en Fc1, vaut-elle pour Fc2/Fc3 ?

SONDE SUPERVISÉE = DIAGNOSTIC, JAMAIS UNE MÉTHODE (DB-01).

vB01 : sonde par unité → l'information de dégradation est présente, linéairement, dans toutes les
classes. Mais chaque sonde avait les étiquettes de SON unité. Ici, cadre supervisé réaliste : une
seule sonde, apprise sur les unités d'ENTRAÎNEMENT (Fc1, mêmes unités que H-CDE, toutes leurs
fenêtres, cible = dégradation vraie), appliquée telle quelle à la référence (Fc1, unité non vue) et
aux unités de classes non vues. Même découpage que H-CDE : comparaison directe.

Deux sondes, déclarées d'avance :
  - lin    : ridge sur capteurs + variables d'opération (moyennes de fenêtre) ;
  - inter  : ridge sur capteurs + W + W² + capteurs × W (la lecture de la dégradation peut dépendre
             du point de fonctionnement, au premier ordre).
Le modèle est unique : T1 ET T2 (seuil de la référence, D-23) ont un sens ici.

Lecture fixée AVANT de mesurer (Spearman principale, par vol, unités non vues, contre la référence) :
  - non vues ≈ référence (Spearman ≥ 0,9) → une lecture commune existe et se transfère ; H-CDE rate
    quelque chose de commun aux classes : le problème est l'apprentissage non supervisé ;
  - non vues au niveau de H-CDE ou en dessous → la lecture dépend du point de fonctionnement et ne
    s'extrapole pas hors du support de Fc1 : même supervisé, le transfert de classe est dur ;
  - si `inter` transfère et pas `lin` (ou l'inverse), c'est la forme de la dépendance aux conditions
    qui compte ; on le note, sans conclure au-delà.

Machine : Mac, ~1 min. Sortie : voie_B_avec_exo/results/vB02_transfert_par_vol.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
from evaluateur import evaluer  # noqa: E402

N_S, N_W = len(v.S_COLS), len(v.W_COLS)


def interactions(X: np.ndarray) -> np.ndarray:
    S, W = X[:, :N_S], X[:, N_S:]
    SW = (S[:, :, None] * W[:, None, :]).reshape(len(X), -1)
    return np.hstack([S, W, W ** 2, SW])


def sondes():
    return {
        "lin": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        # standardiser AVANT les produits (sinon les produits sont dominés par les échelles brutes)
        "inter": make_pipeline(StandardScaler(), FunctionTransformer(interactions), StandardScaler(),
                               Ridge(alpha=1.0)),
    }


def support(f_tr: pd.DataFrame, f_u: pd.DataFrame) -> str:
    """Part des fenêtres de l'unité hors de l'étendue [min, max] de Fc1 d'entraînement, par variable W."""
    return " ".join(f"{c[2:]}:{100 * ((f_u[c] < f_tr[c].min()) | (f_u[c] > f_tr[c].max())).mean():.0f}%"
                    for c in v.W_COLS)


def main():
    it18 = pd.read_csv(v.TRONC / "results/it18_references_par_vol.csv")
    lignes = []
    for ds, key in v.RUNS.items():
        v.it10.DILATION = v.parse_key(key)[1]
        fc, train, ref, unseen = v.PLAN[ds]
        df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
        lab = df.groupby(["unit", "cycle"]).degradation.first().abs()
        f = {u: v.fenetres(df, u).drop(columns="cycle_debut") for u in list(train) + [ref] + list(unseen)}
        f_tr = pd.concat([f[u] for u in train], ignore_index=True)
        y_tr = lab.loc[list(zip(f_tr.unit, f_tr.cycle))].values
        for u in [ref] + list(unseen):
            print(f"{ds} u{u} hors support Fc1 (fenêtres) : {support(f_tr, f[u])}")
        for nom, sonde in sondes().items():
            sonde.fit(f_tr[v.S_COLS + v.W_COLS].values, y_tr)
            fen = pd.concat([x.assign(score=sonde.predict(x[v.S_COLS + v.W_COLS].values))
                             for x in f.values()], ignore_index=True)
            res = evaluer(fen, ds=ds, train_units=train, ref_unit=ref, non_vues=list(unseen))
            lignes.append(res.assign(methode=f"transfert_{nom} (étiquettes)", DS=ds))
    t = pd.concat(lignes, ignore_index=True)
    t.to_csv(v.OUT / "vB02_transfert_par_vol.csv", index=False)

    # --- vue globale unité par unité, contre H-CDE et naïf (médiane des runs valides, it18) ---
    cols = ["AUROC", "corr_spearman", "corr_pearson", "reponse_relative", "FA_classe", "rappel",
            "delai_frac", "non_detecte"]
    h = it18.groupby(["methode", "DS", "unite"])[cols].median().reset_index()
    v_ = pd.concat([t[["methode", "DS", "unite"] + cols], h])
    v_["classe"] = v_.apply(lambda r: f"Fc{v.PLAN[r.DS][3].get(r.unite, 1)}", axis=1)
    pd.set_option("display.width", 220)
    for m in cols:
        print(f"\n=== {m}, par vol ===")
        print(v_.pivot_table(index=["DS", "unite", "classe"], columns="methode", values=m)
              .round(3).to_string())
    print(f"\nécrit : {v.OUT / 'vB02_transfert_par_vol.csv'}")


if __name__ == "__main__":
    main()
