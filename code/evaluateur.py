"""Évaluateur commun aux deux voies (PROTOCOLE.md §4, §6 ; D-20, D-23, D-24).

Une méthode quelconque (voie A, voie B, H-CDE, naïf) fournit UN SCORE PAR FENÊTRE, plus grand = plus
dégradé, avec ses métadonnées :

    unit    : unité
    cycle   : vol où se TERMINE la fenêtre
    score   : score d'anomalie (orienté : plus grand = plus dégradé)
    W_alt, W_Mach, W_TRA, W_T2 : moyennes des conditions de vol sur la fenêtre (ÉVALUATION seulement,
              pour les fausses alarmes intra-unité ; en voie A elles ne servent jamais au modèle)

Les fenêtres des unités d'entraînement doivent être incluses : leurs conditions saines servent de
référence pour « conditions inhabituelles ».

L'évaluateur joint LUI-MÊME les étiquettes `hs` et `degradation` depuis le CSV, par (unit, cycle) :
elles sont constantes pendant un vol (vérifié le 23 sept. sur DS01/04/05/07). Une méthode n'a donc
jamais à manipuler les étiquettes.

Granularité (D-24) :
  - détection et fausses alarmes entre classes : PAR VOL (score du vol = médiane des fenêtres qui se
    terminent dans ce vol) ;
  - fausses alarmes intra-unité (conditions inhabituelles) : PAR FENÊTRE.
`granularite="fenetre"` calcule tout par fenêtre : sert UNIQUEMENT à vérifier que l'évaluateur
reproduit it12 (it18), pas à rapporter des résultats.

Métriques, par unité évaluée (référence + unités non vues), en DEUX TABLEAUX qu'on ne mélange pas
(justification complète : PROTOCOLE.md §4) :

  T1 — SANS SEUIL (comparer les méthodes) :
      AUROC dégradé/sain ; corrélation score/dégradation vraie de SPEARMAN (principale, D-25 : rangs,
      insensible à une transformation monotone, D-20) et de PEARSON (complément, comparable à
      l'alignment R² de H-CDE) ; réponse = (écart de score dégradé − sain) / (écart de dégradation
      vraie), rapportée à celle de la référence ; FPR95 (à 95 % de dégradé détecté, fraction de sain
      au-dessus : seuil propre à l'unité, choisi avec ses étiquettes — convention OOD).

  T2 — AU SEUIL DE LA RÉFÉRENCE (D-23 : quantile 95 % du sain de la référence, appliqué tel quel ;
       fonctionnement réel, question du transfert ; D-25) :
      FA_classe  : fausses alarmes sur les vols SAINS (5 % par construction sur la référence) ;
      rappel     : part des vols DÉGRADÉS au-dessus du seuil ;
      délai      : alarme = premier vol qui ouvre une série de N_PERSISTANCE = 3 vols consécutifs
                   au-dessus du seuil ; délai exprimé en DÉGRADATION ACCUMULÉE à ce vol, pas en vols
                   (Fleet/METRICS_FROZEN.md A2), et en fraction de la dégradation de la vie de l'unité
                   (0 = début de vie, 1 = fin) ; `alarme_avant_apparition` si ce vol est encore sain
                   (alarme soutenue sur le sain) ;
      non_detecte : aucune série de 3 vols ; jamais imputé par une valeur de délai.
      Le rappel et le délai sont indissociables de FA_classe : une méthode peut baisser ses fausses
      alarmes en se désensibilisant (sur-suppression) ; seul le couple les révèle
      (Fleet/METRICS_FROZEN.md §3).

  Hors tableaux — fausses alarmes intra-unité (comme it10), PAR FENÊTRE : fenêtres saines de l'unité ;
      « inhabituelles » = 10 % les plus éloignées des conditions saines d'entraînement ; seuil à 5 %
      sur les autres, lu sur celles-ci.

Non rapportées, et pourquoi (PROTOCOLE.md §4) : précision et F1 (dépendent de la proportion de vols
dégradés, artefact du run-to-failure ; F1 fusionne les deux moitiés du problème) ; corrélation à âge
contrôlé (retirée le 23 sept.).

Pas d'alignment ici : il porte sur un latent, pas sur un score, et ne sert qu'à la comparaison avec le
papier H-CDE (D-20) — il reste calculé par it12.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HCDE = Path(__file__).resolve().parents[1] / "hcde"
DATASETS = HCDE / "run/datasets/ncmapss"
W_COLS = ["W_alt", "W_Mach", "W_TRA", "W_T2"]
Q_SEUIL = 0.95          # 5 % de fausses alarmes sur le sain de référence
Q_INHABITUEL = 0.90     # conditions inhabituelles = 10 % les plus éloignées (it10)
MIN_VOLS = 5            # en dessous, une métrique d'unité n'est pas calculée
N_PERSISTANCE = 3       # vols consécutifs au-dessus du seuil pour déclencher l'alarme (Martin, 23 sept.)


# --------------------------------------------------------------------------------------------------
# briques (identiques à it1/it10, recopiées ici pour que l'évaluateur ne dépende pas de H-CDE)
# --------------------------------------------------------------------------------------------------
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score dégradé > score sain), par les rangs."""
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort()
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 - 1) / 2) / (n1 * n2))


def fpr_at_tpr(pos: np.ndarray, neg: np.ndarray, tpr: float = 0.95) -> float:
    thr = np.quantile(pos, 1 - tpr)
    return float((neg >= thr).mean())


def w_distance(W: np.ndarray, W_ref: np.ndarray) -> np.ndarray:
    mu, sd = W_ref.mean(0), W_ref.std(0) + 1e-9
    return np.linalg.norm((W - mu) / sd, axis=1)


# --------------------------------------------------------------------------------------------------
def etiquettes(ds: str) -> pd.DataFrame:
    """hs et degradation par (unit, cycle), depuis le CSV du protocole."""
    df = pd.read_csv(DATASETS / f"N-CMAPSS_{ds}.csv", usecols=["unit", "cycle", "hs", "degradation"])
    g = df.groupby(["unit", "cycle"])
    assert (g.hs.nunique() == 1).all() and ((g.degradation.max() - g.degradation.min()) < 1e-12).all(), \
        f"{ds} : étiquettes non constantes pendant un vol — la jointure par vol n'est plus valide"
    lab = g.agg(hs=("hs", "first"), degradation=("degradation", "first")).reset_index()
    lab["degradation"] = lab.degradation.abs()
    return lab


def _joindre(fen: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    manquants = {"unit", "cycle", "score"} - set(fen.columns)
    assert not manquants, f"colonnes manquantes : {manquants}"
    interdits = {"hs", "degradation"} & set(fen.columns)
    assert not interdits, f"la méthode ne doit pas fournir d'étiquettes : {interdits}"
    d = fen.merge(lab, on=["unit", "cycle"], how="left", validate="many_to_one")
    assert d.hs.notna().all(), "fenêtres sans vol correspondant dans le CSV"
    return d


def par_vol(d: pd.DataFrame) -> pd.DataFrame:
    """Un score par vol = médiane des fenêtres qui se terminent dans ce vol (D-24)."""
    return (d.groupby(["unit", "cycle"])
             .agg(score=("score", "median"), hs=("hs", "first"),
                  degradation=("degradation", "first"))
             .reset_index())


def alarme(m: pd.DataFrame, seuil: float, n: int = N_PERSISTANCE) -> dict:
    """Premier vol qui ouvre une série de `n` vols consécutifs au-dessus du seuil (vols triés ; un vol
    sans fenêtre n'interrompt pas la série — il n'existe pas dans N-CMAPSS, voir it18)."""
    m = m.sort_values("cycle")
    au_dessus = (m.score.values >= seuil).astype(int)
    serie = np.convolve(au_dessus, np.ones(n, dtype=int), mode="valid") == n
    d0, d1 = m.degradation.iloc[0], m.degradation.iloc[-1]
    if not serie.any():
        return dict(non_detecte=True, vol_alarme=np.nan, delai_degr=np.nan, delai_frac=np.nan,
                    alarme_avant_apparition=False)
    i = int(np.argmax(serie))
    d = float(m.degradation.iloc[i])
    return dict(non_detecte=False, vol_alarme=int(m.cycle.iloc[i]), delai_degr=d,
                delai_frac=float((d - d0) / (d1 - d0)), alarme_avant_apparition=bool(m.hs.iloc[i] == 1))


def evaluer(fen: pd.DataFrame, ds: str, train_units, ref_unit: int, non_vues,
            granularite: str = "vol") -> pd.DataFrame:
    """Retourne une ligne par unité évaluée."""
    assert granularite in ("vol", "fenetre")
    d = _joindre(fen, etiquettes(ds))
    e = par_vol(d) if granularite == "vol" else d          # granularité de la détection
    unites = [ref_unit] + list(non_vues)

    def sain(u):
        return e[(e.unit == u) & (e.hs == 1)].score.values

    seuil_ref = np.quantile(sain(ref_unit), Q_SEUIL)        # D-23

    # référence des conditions : fenêtres saines des unités d'entraînement (it10)
    a_wf = set(W_COLS) <= set(d.columns)
    if a_wf:
        W_ref = d[d.unit.isin(train_units) & (d.hs == 1)][W_COLS].values

    rows = []
    for u in unites:
        m = e[e.unit == u]
        s_h, s_d = m[m.hs == 1], m[m.hs == 0]
        if len(s_h) < MIN_VOLS or len(s_d) < MIN_VOLS:
            print(f"  u{u} : trop peu d'échantillons (sain {len(s_h)}, dégradé {len(s_d)}) — ignorée")
            continue
        ampleur = float(np.median(s_d.degradation) - np.median(s_h.degradation))
        ecart = float(np.median(s_d.score) - np.median(s_h.score))
        r = dict(
            unite=u, role="ref" if u == ref_unit else "non_vue", granularite=granularite,
            n_sain=len(s_h), n_degrade=len(s_d),
            AUROC=auroc(s_d.score.values, s_h.score.values),
            FPR95=fpr_at_tpr(s_d.score.values, s_h.score.values),
            corr_pearson=float(np.corrcoef(m.score, m.degradation)[0, 1]),
            corr_spearman=float(spearmanr(m.score, m.degradation).statistic),
            ampleur_vraie=ampleur, ecart_score=ecart, reponse=ecart / ampleur,
            FA_classe=float((s_h.score.values >= seuil_ref).mean()),
            rappel=float((s_d.score.values >= seuil_ref).mean()),
        )
        if granularite == "vol":                         # le délai n'a de sens que vol par vol
            r.update(alarme(m, seuil_ref))
        # fausses alarmes intra-unité : toujours PAR FENÊTRE (D-24)
        if a_wf:
            h = d[(d.unit == u) & (d.hs == 1)]
            dist = w_distance(h[W_COLS].values, W_ref)
            loin = dist >= np.quantile(dist, Q_INHABITUEL)
            seuil_u = np.quantile(h.score.values[~loin], Q_SEUIL)
            r["FA_intra"] = float((h.score.values[loin] >= seuil_u).mean())
        rows.append(r)
    res = pd.DataFrame(rows)
    if len(res) and (res.role == "ref").any():
        res["reponse_relative"] = res.reponse / res.loc[res.role == "ref", "reponse"].iloc[0]

    return res
