"""vB05c — Le test de dérive lente de l'usage (vB05), refait avec la SFA régularisée (vB06).

vB05 n'était pas interprétable à dérive forte : la SFA à 14 capteurs s'effondre déjà avec ~25 vols par
unité, sans dérive (vB05b). vB06 : la SFA réduite aux composantes expliquant 99 % de la variance
(`sfa_v99`) tient avec ~25 vols sur DS01 (0,94 / 0,92), moins sur DS05 et DS07 (0,79–0,85).

Méthodes, déclarées avant mesure :
  - sans_W_v99       : `sfa_v99` de vB06, capteurs seuls ;
  - insens_W_v99     : directions insensibles aux conditions (B·w = 0, B estimé sur les différences
                       entre vols consécutifs, comme vB05), PUIS réduction à 99 % de variance dans ce
                       sous-espace, PUIS SFA.
Scénarios : ceux de vB05 (aucune, montante h = 0,30, montante h = 0,15, descendante h = 0,15), dérive
dans les unités d'entraînement seulement, unités évaluées complètes. Comparaison à la bonne taille :
vB06 donne `sfa_v99` sur des vols tirés au hasard SANS dérive, aux mêmes tailles (~45, ~25).

Lecture fixée AVANT de mesurer (Spearman par vol, non vues), sur les DS où `sfa_v99` tient à ~25 vols
sans dérive (DS01 ; DS05 et DS07 rapportés mais moins lisibles) :
  - sans_W_v99 sous dérive h = 0,15 nettement sous son contrôle de même taille (vB06, ~25) et R²(score,
    W) qui monte → la lenteur est piégée par la dérive d'usage ;
  - insens_W_v99 proche de son niveau sans dérive → les variables d'opération la protègent (apport de
    la voie B) ;
  - sinon on décrit.

Machine : Mac. Sortie : voie_B_avec_exo/results/vB05c_*.csv, vB05c.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402  (ajoute aussi ../code au chemin)
import vB05_derive_usage as d5  # noqa: E402
import vB06_sfa_regularisee as r6  # noqa: E402
import sfa  # noqa: E402
from evaluateur import evaluer  # noqa: E402


def insens_v99(X, W, unit, cycle) -> sfa.SFA:
    mu, sd = X.mean(0), X.std(0)
    muW, sdW = W.mean(0), W.std(0)
    k, kw = X.shape[1], W.shape[1]
    fv = sfa.moyennes_par_vol(np.hstack([(X - mu) / sd, (W - muW) / sdW]), unit, cycle)
    dS, dW = [], []
    for _, g in fv.sort_values(["unit", "cycle"]).groupby("unit"):
        dS.append(np.diff(g[list(range(k))].values, axis=0))
        dW.append(np.diff(g[list(range(k, k + kw))].values, axis=0))
    B, *_ = np.linalg.lstsq(np.vstack(dW), np.vstack(dS), rcond=None)
    N = np.linalg.svd(B)[2][kw:].T                                    # B·N = 0
    Z = fv[list(range(k))].values @ N
    val, vec = np.linalg.eigh(np.cov(Z, rowvar=False))
    val, vec = val[::-1], vec[:, ::-1]
    q = int(np.searchsorted(np.cumsum(val) / val.sum(), r6.VAR_CIBLE) + 1)
    U = vec[:, :q]
    fz = pd.DataFrame(Z @ U).assign(unit=fv.unit.values, cycle=fv.cycle.values)
    # q = 1 (une seule composante porte 99 % de la variance du sous-espace insensible) : la règle
    # s'applique telle quelle, la direction est cette composante (constaté à la 1re exécution, 24 sept.)
    z = np.ones(1) if q == 1 else sfa.direction_lente(fz, q)
    z = sfa.orienter(fz, q, z)
    INFO.append(q)
    return sfa.SFA(moyenne=mu, ecart_type=sd, w=N @ U @ z)


INFO: list[int] = []                       # q retenus, pour le rapport


def main():
    lignes = []
    for ds in d5.DS_:
        v.it10.DILATION = v.parse_key(ds)[1]
        fc, train, ref, unseen = v.PLAN[ds]
        evals = [ref] + list(unseen)
        df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
        complets = {u: v.fenetres(df, u) for u in evals}
        for nom_sc, derive in d5.SCENARIOS.items():
            tr = pd.concat([d5.fenetres_derive(df, u, derive)[0] for u in train], ignore_index=True)
            args = (tr[v.S_COLS].values, tr.unit.values, tr.cycle.values)
            modeles = {"sans_W_v99": r6.ajuster_reduite(*args)[0],
                       "insens_W_v99": insens_v99(tr[v.S_COLS].values, tr[v.W_COLS].values,
                                                  tr.unit.values, tr.cycle.values)}
            for nom_m, m in modeles.items():
                fen = pd.concat([x.assign(score=m.score(x[v.S_COLS].values))
                                 for x in list(complets.values()) + [tr]], ignore_index=True)
                res = evaluer(fen, ds=ds, train_units=train, ref_unit=ref, non_vues=list(unseen))
                r2 = {}
                for u in evals:
                    msk = (fen.unit == u).values
                    Wu = fen.loc[msk, v.W_COLS].values
                    r2[u] = LinearRegression().fit(Wu, fen.score[msk]).score(Wu, fen.score[msk])
                res["R2_score_W"] = res.unite.map(r2)
                lignes.append(res.assign(DS=ds, scenario=nom_sc, methode=nom_m))
    t = pd.concat(lignes, ignore_index=True)
    t.to_csv(v.OUT / "vB05c_derive_regularisee.csv", index=False)

    # contrôle de même taille, sans dérive (vB06, sfa_v99)
    c6 = pd.read_csv(v.OUT / "vB06_sfa_regularisee.csv")
    c6 = (c6[c6.methode == "sfa_v99"].groupby(["DS", "unite", "taille"]).spearman.median()
          .unstack("taille")[["tout", "~45", "~25"]].add_prefix("hasard_"))
    pd.set_option("display.width", 250)
    print(f"q retenus par insens_W_v99 (DS × scénario) : {INFO}")
    ordre = list(d5.SCENARIOS)
    for role in ["non_vue", "ref"]:
        x = t[t.role == role]
        for mtr in ["corr_spearman", "R2_score_W"]:
            print(f"\n=== {mtr}, {role} ===")
            p = x.pivot_table(index=["DS", "unite"], columns=["methode", "scenario"], values=mtr)
            p = p.reindex(columns=[(a, s) for a in ["sans_W_v99", "insens_W_v99"] for s in ordre])
            p.columns = [f"{a}|{s}" for a, s in p.columns]
            if mtr == "corr_spearman":
                p = p.join(c6)
            print(p.round(2).to_string())


if __name__ == "__main__":
    main()
