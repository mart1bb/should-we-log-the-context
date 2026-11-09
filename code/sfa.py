"""Analyse en caractéristiques lentes (SFA), implémentation commune aux deux voies (D-26).

Origine : voie B, vB04 (`voie_B_avec_exo/code/vB04_direction_lente.py`, 24 sept.). Déplacée dans le
tronc commun pour que les deux voies utilisent exactement la même version.

Idée (Wiskott & Sejnowski 2002) : chercher la combinaison LINÉAIRE des entrées qui varie le moins d'un
vol au suivant, à variance totale fixée. Sur N-CMAPSS, les conditions de vol changent d'un vol à
l'autre et la dégradation dérive lentement : la direction la plus lente suit la dégradation et évite
les effets des conditions (vB04 : Spearman par vol 0,92–0,99 sur les classes non vues de DS01, DS05,
DS07, capteurs seuls ; DS04, fan, inversé).

Ce qui entre dans la méthode (PROTOCOLE §2) :
  - des entrées par fenêtre (capteurs, ou résidus) — aucune étiquette ;
  - l'appartenance d'une fenêtre à un vol et à une unité ;
  - l'ORDRE des vols, dans le critère (différences entre vols consécutifs) et pour choisir le signe
    (score croissant avec l'ordre sur les unités d'entraînement). Jamais l'âge en entrée.

Détails :
  - agrégation : moyenne des entrées sur les fenêtres qui se terminent dans le vol ;
  - blanchiment sur le sous-espace de variance non nulle (valeurs propres > eps × la plus grande) :
    indispensable, les capteurs P15 et P21 sont presque colinéaires (sans lui, la « plus lente » est
    une direction de variance nulle, bug du 24 sept. consigné dans le JOURNAL de la voie B) ;
  - `ecart` : écart entre vols comparés (1 = vols consécutifs) ;
  - robustesse vérifiée (vB04b) : eps 1e-8 / 1e-6 / 1e-4, écart 1 ou 3, avec ou sans P15.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EPS = 1e-6


def moyennes_par_vol(X: np.ndarray, unit: np.ndarray, cycle: np.ndarray) -> pd.DataFrame:
    """Une ligne par (unité, vol) : moyenne des colonnes de X sur les fenêtres de ce vol."""
    d = pd.DataFrame(X).assign(unit=unit, cycle=cycle)
    return d.groupby(["unit", "cycle"]).mean().reset_index()


def rang_dans_la_vie(fv: pd.DataFrame) -> np.ndarray:
    """Position du vol dans la vie de son unité, de 0 à 1 (sert seulement à orienter le signe)."""
    return fv.groupby("unit").cycle.rank(pct=True).values


def direction_lente(fv: pd.DataFrame, k: int, eps: float = EPS, ecart: int = 1) -> np.ndarray:
    """Direction (norme 1) la plus lente des k premières colonnes de `fv` (moyennes par vol)."""
    cols = list(range(k))
    C = np.cov(fv[cols].values, rowvar=False)
    val, vec = np.linalg.eigh(C)
    garde = val > eps * val.max()
    P = vec[:, garde] / np.sqrt(val[garde])                 # blanchiment : Pᵀ C P = I
    dX = np.vstack([g[cols].values[ecart:] - g[cols].values[:-ecart]
                    for _, g in fv.sort_values(["unit", "cycle"]).groupby("unit")])
    _, vecs = np.linalg.eigh(np.cov(dX @ P, rowvar=False))  # croissantes : la 1re est la plus lente
    w = P @ vecs[:, 0]
    return w / np.linalg.norm(w)


def orienter(fv: pd.DataFrame, k: int, w: np.ndarray) -> np.ndarray:
    """Signe choisi pour que le score croisse avec l'ordre des vols (pas avec la dégradation)."""
    s = fv[list(range(k))].values @ w
    return -w if spearmanr(s, rang_dans_la_vie(fv)).statistic < 0 else w


@dataclass
class SFA:
    """SFA ajustée : standardisation des entrées + direction lente orientée."""
    moyenne: np.ndarray
    ecart_type: np.ndarray
    w: np.ndarray

    def score(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.moyenne) / self.ecart_type) @ self.w


def ajuster(X: np.ndarray, unit: np.ndarray, cycle: np.ndarray, eps: float = EPS,
            ecart: int = 1) -> SFA:
    """Ajuste la SFA sur des fenêtres d'ENTRAÎNEMENT (X : une ligne par fenêtre).

    Standardisation sur ces fenêtres (écart-type ddof=0, comme StandardScaler), moyennes par vol,
    direction la plus lente, orientation par l'ordre des vols.
    """
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    fv = moyennes_par_vol(Z, unit, cycle)
    k = X.shape[1]
    w = orienter(fv, k, direction_lente(fv, k, eps, ecart))
    return SFA(moyenne=mu, ecart_type=sd, w=w)


VAR_CIBLE = 0.99


def ajuster_reduite(X: np.ndarray, unit: np.ndarray, cycle: np.ndarray, var_cible: float = VAR_CIBLE,
                    q: int | None = None) -> tuple[SFA, int]:
    """SFA régularisée (`sfa_v99`) : SFA dans l'espace des q premières composantes principales.

    Origine : voie B, vB06 (24 sept.) ; promue au tronc commun à la demande de la voie A (CANAL M-05).
    Même standardisation et mêmes moyennes par vol qu'`ajuster` ; ACP des moyennes par vol
    d'entraînement ; q = plus petit nombre de composantes expliquant ≥ `var_cible` de la variance (ou
    `q` imposé) ; SFA dans cet espace, orientation par l'ordre des vols. Si q = 1, la direction est
    cette composante (la SFA n'a pas de choix).
    vB06 : avec ~25 vols par unité d'entraînement (sans dérive), Spearman non vues 0,79–0,94 contre
    0,42–0,76 pour `ajuster` ; petit coût avec tous les vols (DS05 u9 0,84 contre 0,92).
    Entrées en float64 (CANAL M-01) : une moyenne de fenêtre calculée en float32 fait bouger la SFA.
    Retourne le modèle et le q retenu.
    """
    mu, sd = X.mean(0), X.std(0)
    fv = moyennes_par_vol((X - mu) / sd, unit, cycle)
    k = X.shape[1]
    Z = fv[list(range(k))].values
    val, vec = np.linalg.eigh(np.cov(Z, rowvar=False))
    val, vec = val[::-1], vec[:, ::-1]                            # décroissantes
    if q is None:
        q = int(np.searchsorted(np.cumsum(val) / val.sum(), var_cible) + 1)
    U = vec[:, :q]
    fz = pd.DataFrame(Z @ U).assign(unit=fv.unit.values, cycle=fv.cycle.values)
    z = np.ones(1) if q == 1 else direction_lente(fz, q)
    z = orienter(fz, q, z)
    return SFA(moyenne=mu, ecart_type=sd, w=U @ z), q
