"""vB05 — Dérive lente de l'usage : la lenteur se trompe-t-elle, et les variables d'opération la protègent-elles ?

Question de la voie B (D-26) : qu'apportent les variables d'opération quand la lenteur ne suffit plus ?
La SFA sur capteurs seuls (voie A ; vB04) suppose que la dégradation est la source la plus lente. Sur
N-CMAPSS, les conditions de vol ne dérivent pas avec l'âge (vérifié : |ρ(altitude moyenne du vol, âge)|
≤ 0,24 dans toutes les unités du protocole). Sur un robot, l'usage peut dériver lentement. On FABRIQUE
cette dérive et on regarde qui y résiste.

Fabrication (déclarée avant mesure) : dans une unité, a = rang (0–1) de l'altitude moyenne du vol, t =
rang (0–1) du vol dans la vie. On garde les vols où |a − t| < h (dérive montante : altitude qui croît
avec l'âge) ou |a − (1 − t)| < h (descendante). La vie de l'unité devient la suite des vols gardés,
dans l'ordre ; les fenêtres sont recalculées sur cette suite (elles ne mélangent pas de vols écartés).
Les étiquettes restent celles des vols gardés. Scénarios : aucune dérive ; montante h = 0,3 ;
montante h = 0,15 ; descendante h = 0,15.
  - PRINCIPAL : dérive dans les unités d'ENTRAÎNEMENT seulement ; unités évaluées complètes (la
    direction apprise est-elle corrompue ?).
  - SECONDAIRE : même dérive aussi dans les unités évaluées (l'unité surveillée dérive elle-même).

Méthodes, déclarées avant mesure (aucune étiquette ; ordre des vols dans le critère) :
  - sans_W_sfa    : SFA commune (`../code/sfa.py`) sur les 14 capteurs — la méthode de la voie A ;
  - cond_lin_sfa  : SFA sur le résidu d'une régression linéaire capteurs ~ W ajustée sur les NIVEAUX
                    (comme vB04) ;
  - sfa_insens_W  : NOUVELLE. On estime la réponse des capteurs aux conditions à partir des
                    DIFFÉRENCES entre vols consécutifs d'une même unité (ΔS ≈ ΔW·B, moindres carrés,
                    sans constante) : d'un vol au suivant, la dégradation bouge très peu et une dérive
                    lente aussi, donc ces différences isolent l'effet des conditions même quand l'usage
                    dérive. On restreint ensuite la SFA aux directions w insensibles aux conditions
                    (B·w = 0) ; le score reste wᵀS, sans rien soustraire (pas de modèle de condition à
                    extrapoler, contrairement à cond_lin_sfa).
DS : DS01, DS05, DS07 (DS04 exclu : direction de dégradation hétérogène entre unités, vB04c).

Lecture fixée AVANT de mesurer (Spearman par vol, unités non vues ; R² du score sur W dans ces unités) :
  - contrôle : « aucune dérive » doit redonner vB04 (sans_W_sfa) ;
  - si sans_W_sfa chute avec la dérive (Spearman non vues < 0,8, ou R²(score, W) qui monte) ET
    sfa_insens_W reste ≥ 0,9 → les variables d'opération protègent la lenteur contre une dérive
    lente de l'usage : c'est l'apport de la voie B ;
  - si les deux chutent → l'insensibilité linéaire ne suffit pas ;
  - si sans_W_sfa ne chute pas → la SFA résiste à une dérive de cette ampleur ; on le rapporte, et une
    dérive plus forte serait une AUTRE expérience, décidée avec Martin (pas d'ajustement après coup).

Machine : Mac, quelques minutes. Sortie : voie_B_avec_exo/results/vB05_*.csv, vB05.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402  (ajoute aussi ../code au chemin)
import sfa  # noqa: E402
from evaluateur import evaluer  # noqa: E402

DS_ = ["DS01", "DS05", "DS07"]
SCENARIOS = {"aucune": None, "montante_h0.30": (+1, 0.30), "montante_h0.15": (+1, 0.15),
             "descendante_h0.15": (-1, 0.15)}


def vols_gardes(df_u: pd.DataFrame, derive) -> np.ndarray:
    """Numéros des vols gardés dans une unité pour un scénario de dérive."""
    fv = df_u.groupby("cycle").alt.mean()
    if derive is None:
        return fv.index.values
    sens, h = derive
    a = fv.rank(pct=True).values
    t = pd.Series(fv.index.values).rank(pct=True).values
    cible = t if sens > 0 else 1 - t
    return fv.index.values[np.abs(a - cible) < h]


def fenetres_derive(df: pd.DataFrame, u: int, derive) -> tuple[pd.DataFrame, float, int]:
    d = df[df.unit == u]
    garde = vols_gardes(d, derive)
    d = d[d.cycle.isin(garde)]
    fv = d.groupby("cycle").alt.mean()
    rho = spearmanr(fv.values, fv.index.values).statistic
    return v.fenetres(d, u), float(rho), len(garde)


def sfa_insensible(X: np.ndarray, W: np.ndarray, unit, cycle) -> sfa.SFA:
    """SFA restreinte aux directions insensibles aux conditions (B·w = 0), B estimé sur les différences."""
    mu, sd = X.mean(0), X.std(0)
    muW, sdW = W.mean(0), W.std(0)
    Z, ZW = (X - mu) / sd, (W - muW) / sdW
    k, kw = Z.shape[1], ZW.shape[1]
    fv = sfa.moyennes_par_vol(np.hstack([Z, ZW]), unit, cycle)
    dS, dW = [], []
    for _, g in fv.sort_values(["unit", "cycle"]).groupby("unit"):
        dS.append(np.diff(g[list(range(k))].values, axis=0))
        dW.append(np.diff(g[list(range(k, k + kw))].values, axis=0))
    B, *_ = np.linalg.lstsq(np.vstack(dW), np.vstack(dS), rcond=None)      # (kw × k)
    _, _, Vt = np.linalg.svd(B)
    N = Vt[kw:].T                                                          # (k × (k − kw)), B·N = 0
    fz = pd.DataFrame(fv[list(range(k))].values @ N).assign(unit=fv.unit.values, cycle=fv.cycle.values)
    kz = N.shape[1]
    z = sfa.orienter(fz, kz, sfa.direction_lente(fz, kz))
    return sfa.SFA(moyenne=mu, ecart_type=sd, w=N @ z)


def cond_lin(tr: pd.DataFrame) -> tuple:
    """Résidu d'une régression capteurs ~ W sur les niveaux (comme vB04), puis SFA commune."""
    mu, sd = tr[v.S_COLS].values.mean(0), tr[v.S_COLS].values.std(0)
    muW, sdW = tr[v.W_COLS].values.mean(0), tr[v.W_COLS].values.std(0)
    S = (tr[v.S_COLS].values - mu) / sd
    W = (tr[v.W_COLS].values - muW) / sdW
    f = LinearRegression().fit(W, S)
    m = sfa.ajuster(S - f.predict(W), tr.unit.values, tr.cycle.values)

    def score(x):
        Sx = (x[v.S_COLS].values - mu) / sd
        Wx = (x[v.W_COLS].values - muW) / sdW
        return m.score(Sx - f.predict(Wx))
    return score


def main():
    lignes, derives = [], []
    for ds in DS_:
        v.it10.DILATION = v.parse_key(ds)[1]
        fc, train, ref, unseen = v.PLAN[ds]
        evals = [ref] + list(unseen)
        df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
        complets = {u: v.fenetres(df, u) for u in evals}
        for nom_sc, derive in SCENARIOS.items():
            tr_parts = []
            for u in train:
                x, rho, n = fenetres_derive(df, u, derive)
                tr_parts.append(x)
                derives.append(dict(DS=ds, scenario=nom_sc, unite=u, role="train", vols=n,
                                    rho_alt_age=round(rho, 2)))
            tr = pd.concat(tr_parts, ignore_index=True)
            m_sans = sfa.ajuster(tr[v.S_COLS].values, tr.unit.values, tr.cycle.values)
            m_ins = sfa_insensible(tr[v.S_COLS].values, tr[v.W_COLS].values, tr.unit.values,
                                   tr.cycle.values)
            modeles = {                                    # toutes : fenêtres (DataFrame) → score
                "sans_W_sfa": lambda x, m=m_sans: m.score(x[v.S_COLS].values),
                "cond_lin_sfa": cond_lin(tr),
                "sfa_insens_W": lambda x, m=m_ins: m.score(x[v.S_COLS].values),
            }
            # unités évaluées : complètes (principal) ou dérivées elles aussi (secondaire)
            ev_derive = {}
            for u in evals:
                x, rho, n = fenetres_derive(df, u, derive)
                ev_derive[u] = x
                derives.append(dict(DS=ds, scenario=nom_sc, unite=u, role="eval", vols=n,
                                    rho_alt_age=round(rho, 2)))
            for cas, ev in [("principal", complets), ("secondaire", ev_derive)]:
                if cas == "secondaire" and derive is None:
                    continue
                for nom_m, score in modeles.items():
                    parts = [x.assign(score=score(x)) for x in list(ev.values()) + [tr]]
                    fen = pd.concat(parts, ignore_index=True)
                    res = evaluer(fen, ds=ds, train_units=train, ref_unit=ref, non_vues=list(unseen))
                    r2 = {}
                    for u in evals:
                        m = (fen.unit == u).values
                        Wu = fen.loc[m, v.W_COLS].values
                        r2[u] = LinearRegression().fit(Wu, fen.score[m]).score(Wu, fen.score[m])
                    res["R2_score_W"] = res.unite.map(r2)
                    lignes.append(res.assign(DS=ds, scenario=nom_sc, cas=cas, methode=nom_m))
    t = pd.concat(lignes, ignore_index=True)
    t.to_csv(v.OUT / "vB05_derive_par_vol.csv", index=False)
    dv = pd.DataFrame(derives)
    dv.to_csv(v.OUT / "vB05_derives_fabriquees.csv", index=False)

    pd.set_option("display.width", 250)
    print("=== dérive fabriquée : ρ(altitude moyenne du vol, âge) et vols gardés, unités d'entraînement ===")
    print(dv[dv.role == "train"].pivot_table(index=["DS", "unite"], columns="scenario",
                                              values=["rho_alt_age", "vols"]).to_string())
    ordre = list(SCENARIOS)
    for cas in ["principal", "secondaire"]:
        x = t[(t.cas == cas) & (t.role == "non_vue")]
        for m in ["corr_spearman", "R2_score_W", "FA_classe", "rappel"]:
            print(f"\n=== {cas} — {m}, unités NON VUES (par vol) ===")
            p = x.pivot_table(index=["DS", "unite"], columns=["methode", "scenario"], values=m)
            p = p.reindex(columns=[(a, s) for a in ["sans_W_sfa", "cond_lin_sfa", "sfa_insens_W"]
                                   for s in ordre if (a, s) in p.columns])
            print(p.round(2).to_string())
    x = t[(t.cas == "principal") & (t.role == "ref")]
    print("\n=== principal — corr_spearman, RÉFÉRENCE (Fc1) ===")
    print(x.pivot_table(index=["DS"], columns=["methode", "scenario"], values="corr_spearman")
          .round(2).to_string())


if __name__ == "__main__":
    main()
