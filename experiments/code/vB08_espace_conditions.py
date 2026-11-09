"""vB08 — Méthode de la voie B : l'ESPACE des effets des conditions, estimé avec W, sans extrapoler la position.

Contexte :
  - retirer les conditions par un modèle (capteurs ~ W, puis résidu) échoue hors support : il faut
    extrapoler la POSITION où une nouvelle condition place les capteurs (vB03 ; prop. 2 de la voie A) ;
  - la SFA sans W trouve seule une direction insensible aux conditions, car elles changent d'un vol à
    l'autre ; une dérive lente de l'usage la piège (vB05c) ;
  - `insens_W` (vB05, vB05c) estime avec W la partie LINÉAIRE de l'effet des conditions sur les
    différences entre vols consécutifs, et contraint la SFA à y être insensible : résiste à une dérive
    modérée, pas forte. Deux hypothèses : (a) sous dérive forte il ne reste plus assez de variation
    rapide de W pour estimer quoi que ce soit ; (b) les effets NON linéaires des conditions, hors de
    l'espace linéaire estimé, redeviennent la source la plus lente.

Méthode `span_W`, déclarée avant mesure (aucune étiquette ; ordre des vols dans la perte) :
  1. moyennes par vol des capteurs S et de W, standardisés sur les fenêtres d'entraînement ;
  2. caractéristiques non linéaires φ(W) = [W, W², W_i·W_j] (14), standardisées ;
  3. RÉGRESSION SUR LES DIFFÉRENCES entre vols consécutifs d'une même unité : ΔS ≈ Δφ·C (ridge, α = 1).
     Tout ce qui est lent (dégradation, dérive d'usage, constantes d'unité) disparaît des différences :
     C décrit l'effet des conditions seul, y compris non linéaire ;
  4. ESPACE des effets : positions prédites P = φ(W)·C sur les vols d'entraînement, ACP ; V = les r
     premières directions (r : 99 % de la variance de P) ;
  5. SFA régularisée (règle des 99 %, `../code/sfa.py`) dans le complément orthogonal de V ;
  6. score = wᵀS : on n'extrapole jamais la position (on ne soustrait rien).
Comparées : `sans_W_v99` (SFA régularisée sans W, la méthode de la voie A, repère) et `insens_W_v99`
(vB05c). Scénarios : ceux de vB05 (aucune dérive ; montante h = 0,30 ; montante h = 0,15 ; descendante
h = 0,15) dans les unités d'entraînement, unités évaluées complètes. Contrôle de taille pour `span_W` :
vols tirés au hasard sans dérive, tailles de h = 0,30 et h = 0,15 (5 tirages).
Diagnostics : r ; R² des différences (part de ΔS expliquée par Δφ·C) ; angle principal maximal entre
l'espace V estimé sous dérive et sans dérive (stabilité de l'espace).
DS : DS01, DS05, DS07.

Lecture fixée AVANT de mesurer (Spearman par vol, unités non vues) :
  - sans dérive : `span_W` à ± 0,05 de `sans_W_v99` → la contrainte ne coûte rien ;
  - dérive forte : `span_W` ≥ 0,8 là où `sans_W_v99` ≤ 0,1 → la méthode tient ; l'hypothèse (b) était
    la bonne ;
  - dérive forte, `span_W` échoue : si l'espace V reste stable (angle < 20°) → la cause n'est pas
    l'estimation de l'espace, à chercher ailleurs ; si V bouge (angle > 45°) → hypothèse (a) : limite
    d'identifiabilité (plus assez de variation rapide), pas un défaut de modèle.
  - dérive modérée : `span_W` ≥ `insens_W_v99`.

Machine : Mac. Sortie : voie_B_avec_exo/results/vB08_*.csv, vB08.txt
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402  (ajoute aussi ../code au chemin)
import vB05_derive_usage as d5  # noqa: E402
import vB05c_derive_regularisee as d5c  # noqa: E402
import sfa  # noqa: E402
from evaluateur import evaluer  # noqa: E402

N_TIRAGES = 5


def phi(W: np.ndarray) -> np.ndarray:
    if W.shape[1] < 2:                     # une seule variable : pas de produits croisés (vB30)
        return np.hstack([W, W ** 2])
    prod = np.stack([W[:, i] * W[:, j] for i, j in combinations(range(W.shape[1]), 2)], 1)
    return np.hstack([W, W ** 2, prod])


def sfa_dans(Zf: pd.DataFrame, k: int, N: np.ndarray) -> np.ndarray:
    """SFA régularisée (99 %) dans le sous-espace de base N ; renvoie w dans l'espace des capteurs."""
    Z = Zf[list(range(k))].values @ N
    val, vec = np.linalg.eigh(np.cov(Z, rowvar=False))
    val, vec = val[::-1], vec[:, ::-1]
    q = int(np.searchsorted(np.cumsum(val) / val.sum(), sfa.VAR_CIBLE) + 1)
    U = vec[:, :q]
    fz = pd.DataFrame(Z @ U).assign(unit=Zf.unit.values, cycle=Zf.cycle.values)
    z = np.ones(1) if q == 1 else sfa.direction_lente(fz, q)
    return N @ U @ sfa.orienter(fz, q, z)


def sfa_complement(Zf: pd.DataFrame, k: int, N: np.ndarray) -> np.ndarray:
    """SFA NON réduite dans le sous-espace de base N (blanchiment standard de `sfa.direction_lente`)."""
    Z = Zf[list(range(k))].values @ N
    fz = pd.DataFrame(Z).assign(unit=Zf.unit.values, cycle=Zf.cycle.values)
    q = Z.shape[1]
    if q == 0:                                    # les conditions occupent tout l'espace des capteurs (vB31)
        return np.full(k, np.nan)
    z = np.ones(1) if q == 1 else sfa.direction_lente(fz, q)          # complément à une dimension (vB31)
    return N @ sfa.orienter(fz, q, z)


def span_W(X: np.ndarray, W: np.ndarray, unit, cycle, reduire: bool = True) -> tuple[sfa.SFA, dict]:
    """`reduire=True` : version d'origine (vB08, réduction à 99 % dans le complément) ;
    `reduire=False` : version 2 (DB-08, 25 sept.), SFA non réduite dans le complément — la réduction à
    99 % peut écarter la dégradation (vB09b, vB19b)."""
    mu, sd = X.mean(0), X.std(0)
    muW, sdW = W.mean(0), W.std(0)
    k = X.shape[1]
    F = phi((W - muW) / sdW)
    muF, sdF = F.mean(0), F.std(0)
    kf = F.shape[1]
    fv = sfa.moyennes_par_vol(np.hstack([(X - mu) / sd, (F - muF) / sdF]), unit, cycle)
    dS, dF = [], []
    for _, g in fv.sort_values(["unit", "cycle"]).groupby("unit"):
        dS.append(np.diff(g[list(range(k))].values, axis=0))
        dF.append(np.diff(g[list(range(k, k + kf))].values, axis=0))
    dS, dF = np.vstack(dS), np.vstack(dF)
    reg = Ridge(alpha=1.0, fit_intercept=False).fit(dF, dS)
    r2_diff = float(reg.score(dF, dS))
    P = fv[list(range(k, k + kf))].values @ reg.coef_.T               # positions prédites (vols)
    P = P - P.mean(0)
    val, vec = np.linalg.eigh(np.cov(P, rowvar=False))
    val, vec = val[::-1], vec[:, ::-1]
    r = int(np.searchsorted(np.cumsum(val) / val.sum(), sfa.VAR_CIBLE) + 1)
    V = vec[:, :r]
    N = np.linalg.svd(np.eye(k) - V @ V.T)[0][:, :k - r]              # complément orthogonal de V
    w = sfa_dans(fv, k, N) if reduire else sfa_complement(fv, k, N)
    return sfa.SFA(moyenne=mu, ecart_type=sd, w=w), dict(r=r, R2_diff=r2_diff, V=V)


def evaluer_modele(m, complets, tr, ds, train, ref, unseen):
    fen = pd.concat([x.assign(score=m.score(x[v.S_COLS].values)) for x in list(complets.values()) + [tr]],
                    ignore_index=True)
    return evaluer(fen, ds=ds, train_units=train, ref_unit=ref, non_vues=list(unseen))


def main():
    rng = np.random.default_rng(0)
    lignes, diag = [], []
    for ds in d5.DS_:
        v.it10.DILATION = v.parse_key(ds)[1]
        fc, train, ref, unseen = v.PLAN[ds]
        df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
        complets = {u: v.fenetres(df, u) for u in [ref] + list(unseen)}
        V0 = None
        for nom_sc, derive in d5.SCENARIOS.items():
            tr = pd.concat([d5.fenetres_derive(df, u, derive)[0] for u in train], ignore_index=True)
            X, W = tr[v.S_COLS].values, tr[v.W_COLS].values
            m_span, info = span_W(X, W, tr.unit.values, tr.cycle.values)
            if nom_sc == "aucune":
                V0 = info["V"]
            angle = float(np.degrees(subspace_angles(V0, info["V"]).max())) \
                if V0.shape[1] and info["V"].shape[1] else np.nan
            diag.append(dict(DS=ds, scenario=nom_sc, r=info["r"], R2_diff=round(info["R2_diff"], 3),
                             angle_max_vs_aucune=round(angle, 1)))
            modeles = {"span_W": m_span,
                       "insens_W_v99": d5c.insens_v99(X, W, tr.unit.values, tr.cycle.values),
                       "sans_W_v99": sfa.ajuster_reduite(X, tr.unit.values, tr.cycle.values)[0]}
            for nom_m, m in modeles.items():
                res = evaluer_modele(m, complets, tr, ds, train, ref, unseen)
                lignes.append(res.assign(DS=ds, scenario=nom_sc, methode=nom_m, tirage=-1))
        # contrôle de taille pour span_W (sans dérive, vols au hasard)
        for nom_t, sc in [("hasard_~45", "montante_h0.30"), ("hasard_~25", "montante_h0.15")]:
            for k in range(N_TIRAGES):
                parts = []
                for u in train:
                    du = df[df.unit == u]
                    n = len(d5.vols_gardes(du, d5.SCENARIOS[sc]))
                    vols = np.sort(rng.choice(du.cycle.unique(), n, replace=False))
                    parts.append(v.fenetres(du[du.cycle.isin(vols)], u))
                tr = pd.concat(parts, ignore_index=True)
                m, _ = span_W(tr[v.S_COLS].values, tr[v.W_COLS].values, tr.unit.values, tr.cycle.values)
                res = evaluer_modele(m, complets, tr, ds, train, ref, unseen)
                lignes.append(res.assign(DS=ds, scenario=nom_t, methode="span_W", tirage=k))
    t = pd.concat(lignes, ignore_index=True)
    t.to_csv(v.OUT / "vB08_espace_conditions.csv", index=False)
    dg = pd.DataFrame(diag)
    dg.to_csv(v.OUT / "vB08_diagnostics.csv", index=False)

    pd.set_option("display.width", 250)
    print("=== diagnostics span_W : r, R² des différences, angle max. de l'espace V contre « aucune » ===")
    print(dg.to_string(index=False))
    ordre = list(d5.SCENARIOS) + ["hasard_~45", "hasard_~25"]
    for role in ["non_vue", "ref"]:
        x = t[t.role == role]
        print(f"\n=== Spearman par vol, {role} (contrôles de taille : médiane des tirages) ===")
        p = x.pivot_table(index=["DS", "unite"], columns=["methode", "scenario"], values="corr_spearman",
                          aggfunc="median")
        cols = [(a, s) for a in ["sans_W_v99", "insens_W_v99", "span_W"] for s in ordre if (a, s) in p.columns]
        p = p[cols]
        p.columns = [f"{a}|{s}" for a, s in p.columns]
        print(p.round(2).T.to_string())
    x = t[(t.role == "non_vue") & (t.tirage == -1)]
    for mtr in ["FA_classe", "rappel"]:
        print(f"\n=== {mtr}, non vues ===")
        p = x.pivot_table(index=["DS", "unite"], columns=["methode", "scenario"], values=mtr)
        p.columns = [f"{a}|{s}" for a, s in p.columns]
        print(p.round(2).T.to_string())


if __name__ == "__main__":
    main()
