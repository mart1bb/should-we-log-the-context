"""vB23 — Estimer SANS étiquettes la part de la signature de dégradation hors de l'espace des conditions.

Pourquoi : vB22/vB22b — la méthode qui marche dépend de cette part (N-CMAPSS 49–81 % hors → la lenteur
suffit ; batteries 11–26 % → seul un modèle de la position attendue voit l'usure). Mesurée jusqu'ici avec
la vérité de dégradation (DB-01) : inutilisable pour choisir une méthode sur une flotte réelle.

Estimation, déclarée avant mesure (moyennes par vol / cycle, capteurs S standardisés, unités d'entraînement) :
  - espace des conditions V : celui de `span_W` (effet des conditions estimé sur les DIFFÉRENCES entre vols
    consécutifs, donc insensible à la dégradation et à toute dérive lente ; ACP 99 % des positions
    prédites) ; C = ses coefficients ;
  - résidu à conditions égales : R = S − φ(W)·C, centré par unité (le modèle des différences donne l'effet
    des conditions ; la constante propre à l'unité est retirée) ;
  - signature estimée â, deux estimateurs (ordre des vols dans le critère, aucune étiquette) :
      `tendance`  : moyenne sur les unités de la pente de R contre le rang du vol dans la vie (0 → 1) ;
      `sfa_resid` : chargement du score SFA appris sur R : â = Cov(R, s) / Var(s) ;
  - part hors = ‖â − V Vᵀ â‖² / ‖â‖².
Validation :
  - banc (générateur de vB10, niveau calibré de vB11/vB12 : se = 0,02 ; λ = 1 ; σ_f = 1 ; sans dérive), la
    signature étant réglée à une part f_out HORS de l'espace des effets des conditions : f_out ∈ {0,05 ; 0,2 ;
    0,5 ; 0,8 ; 0,95}, 10 graines ; vérité = f_out (et la mesure avec étiquettes de vB22b, pour comparer) ;
  - réel, contre la mesure avec étiquettes (vB22b) : batteries uniformes (par cycle), N-CMAPSS DS01/05/07
    (moyennes brutes par vol). C-MAPSS : estimation seule (pas de vérité de dégradation). XJTU-SY : non
    applicable (W constant par unité : pas de différences en W).

Lecture fixée AVANT de mesurer :
  - banc : Spearman(estimation, f_out) ≥ 0,9 et erreur absolue médiane ≤ 0,15 → l'estimateur marche ;
  - réel : même ordre que la mesure avec étiquettes (batteries < DS05 < DS07, DS01) et écart ≤ 0,2 ;
  - on retient l'estimateur qui satisfait les deux ; si aucun, on le dit.

Machine : Mac. Sortie : voie_B_avec_exo/results/vB23_*.csv, vB23.txt
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB08_espace_conditions as v8  # noqa: E402
import vB10_banc_realiste as b10  # noqa: E402
import vB12_diagramme_phase as p12  # noqa: E402
import vB19_cmapss_avec_w as b19  # noqa: E402
import vB22_batteries as b22  # noqa: E402
import vB22b_signature as s22  # noqa: E402
import sfa  # noqa: E402

F_OUT = [0.05, 0.2, 0.5, 0.8, 0.95]


def estimer(S: np.ndarray, W: np.ndarray, unit, cycle) -> dict:
    """Part hors de l'espace des conditions de la signature estimée (deux estimateurs), sans étiquettes."""
    muS, sdS = S.mean(0), S.std(0)
    muW, sdW = W.mean(0), W.std(0)
    Wz = (W - muW) / sdW
    F = v8.phi(Wz)
    muF, sdF = F.mean(0), F.std(0) + 1e-12
    k, kf = S.shape[1], F.shape[1]
    fv = sfa.moyennes_par_vol(np.hstack([(S - muS) / sdS, (F - muF) / sdF]), unit, cycle)
    fv = fv.sort_values(["unit", "cycle"]).reset_index(drop=True)
    Sv, Fv = fv[list(range(k))].values, fv[list(range(k, k + kf))].values
    dS, dF = [], []
    for _, g in fv.groupby("unit"):
        dS.append(np.diff(g[list(range(k))].values, axis=0))
        dF.append(np.diff(g[list(range(k, k + kf))].values, axis=0))
    from sklearn.linear_model import Ridge
    C = Ridge(alpha=1.0, fit_intercept=False).fit(np.vstack(dF), np.vstack(dS)).coef_.T   # (kf × k)
    P = Fv @ C
    val, vec = np.linalg.eigh(np.cov(P - P.mean(0), rowvar=False))
    val, vec = val[::-1], vec[:, ::-1]
    r = int(np.searchsorted(np.cumsum(val) / val.sum(), sfa.VAR_CIBLE) + 1)
    V = vec[:, :r]
    R = Sv - P
    R = R - pd.DataFrame(R).groupby(fv.unit.values).transform("mean").values
    rang = fv.groupby("unit").cycle.rank(pct=True).values
    pentes = [np.polyfit(rang[m], R[m], 1)[0] for m in [(fv.unit == u).values for u in fv.unit.unique()]]
    a_t = np.mean(pentes, axis=0)
    fr = pd.DataFrame(R).assign(unit=fv.unit.values, cycle=fv.cycle.values)
    w = sfa.orienter(fr, k, sfa.direction_lente(fr, k))
    s = R @ w
    a_s = (R * (s - s.mean())[:, None]).mean(0) / s.var()

    def hors(a):
        h = a - V @ (V.T @ a)
        return float(h @ h / (a @ a))
    return dict(tendance=hors(a_t), sfa_resid=hors(a_s), r=r)


def generer_fout(graine, f_out):
    """Générateur de vB10 (niveau calibré) avec une part f_out de la signature hors de l'espace des effets."""
    nz = dict(p12.CAL)
    rng0 = np.random.default_rng(10_000 + graine)
    # on reproduit b10.generer mais avec la signature réglée : on patche A0 via un générateur local
    K, RC, KW = b10.K, b10.RC, b10.KW
    rng = np.random.default_rng(graine)
    kc = KW + 1
    B = np.linalg.qr(rng.normal(size=(K, RC)))[0]
    M1 = rng.normal(size=(RC, kc)) / np.sqrt(kc)
    M2 = rng.normal(size=(RC, kc * (kc + 1) // 2)) / np.sqrt(kc * (kc + 1) // 2)
    e2 = rng.normal(size=KW); e2 /= np.linalg.norm(e2)
    dedans = B @ rng.normal(size=RC); dedans /= np.linalg.norm(dedans)
    dehors = (np.eye(K) - B @ B.T) @ rng.normal(size=K); dehors /= np.linalg.norm(dehors)
    A0 = np.sqrt(1 - f_out) * dedans + np.sqrt(f_out) * dehors
    t = np.linspace(0, 1, b10.N_VOLS)
    parts = []
    for u, decal in [(1, 0.0), (2, 0.0), (3, 0.0)]:
        c = decal * e2 + 1.0 * rng.normal(size=(b10.N_VOLS, KW))
        cf = np.hstack([c, nz["sh"] * rng.normal(size=(b10.N_VOLS, 1))])
        h = cf @ M1.T + b10.LAM * b10.psi(cf / 2) @ M2.T
        pert = rng0.normal(size=K); pert /= np.linalg.norm(pert)
        A_u = A0 + nz["het"] * pert
        A_u = nz["a"] * A_u / np.linalg.norm(A_u)
        d = rng.uniform(0, 0.1) + 0.3 * t + rng.uniform(0.5, 1.5) * t ** 4 + nz["sm"] * rng.normal(size=b10.N_VOLS)
        x = h @ B.T + np.outer(d, A_u) + rng.normal(scale=nz["se"], size=(b10.N_VOLS, K))
        W_obs = c + nz["sW"] * rng.normal(size=c.shape)
        parts.append(pd.DataFrame(x, columns=b10.S_COLS).join(pd.DataFrame(W_obs, columns=b10.W_COLS))
                     .assign(unit=u, cycle=np.arange(b10.N_VOLS), degradation=d))
    return pd.concat(parts, ignore_index=True)


def main():
    rows = []
    for f_out in F_OUT:
        for g in range(b10.N_GRAINES):
            df = generer_fout(g, f_out)
            est = estimer(df[b10.S_COLS].values, df[b10.W_COLS].values, df.unit.values, df.cycle.values)
            lab = s22.part_hors(df[b10.S_COLS].values, df[b10.W_COLS].values, df.degradation.values,
                                df.unit.values)[0]
            rows.append(dict(lieu="banc", f_out=f_out, graine=g, etiquettes=lab, **est))
    bat = b22.charger()
    tr = bat[bat.bat.isin(b22.GROUPES["uniforme"])]
    fv = tr.groupby(["bat", "cycle"])[b22.S_B + b22.W_B + ["capacite"]].mean().reset_index()
    est = estimer(fv[b22.S_B].values, fv[b22.W_B].values, fv.bat.values, fv.cycle.values)
    lab = s22.part_hors(fv[b22.S_B].values, fv[b22.W_B].values, -fv.capacite.values, fv.bat.values)[0]
    rows.append(dict(lieu="batteries", etiquettes=lab, **est))
    for ds in ["DS01", "DS05", "DS07"]:
        fc, train, ref, unseen = v.PLAN[ds]
        d = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
        d = d[d.unit.isin(train)]
        sc, wc = [c[2:] for c in v.S_COLS], [c[2:] for c in v.W_COLS]
        f = d.groupby(["unit", "cycle"])[sc + wc + ["degradation"]].mean().reset_index()
        S, W = f[sc].values.astype(float), f[wc].values.astype(float)
        est = estimer(S, W, f.unit.values, f.cycle.values)
        lab = s22.part_hors(S, W, f.degradation.abs().values, f.unit.values)[0]
        rows.append(dict(lieu=f"N-CMAPSS {ds}", etiquettes=lab, **est))
    for fd in ["FD002", "FD004"]:
        d = b19.a6.charger(fd)
        X = [c for c in b19.a6.COLS[5:] if d[c].std() > 1e-8]
        est = estimer(d[X].values.astype(float), d[b19.W_C].values.astype(float), d.unit.values, d.cycle.values)
        rows.append(dict(lieu=f"C-MAPSS {fd}", etiquettes=np.nan, **est))
    t = pd.DataFrame(rows)
    t.to_csv(v.OUT / "vB23_part_hors.csv", index=False)
    pd.set_option("display.width", 200)
    banc = t[t.lieu == "banc"]
    print("=== banc : médianes par f_out (vérité) ===")
    print(banc.groupby("f_out")[["etiquettes", "tendance", "sfa_resid", "r"]].median().round(2).to_string())
    for est in ["tendance", "sfa_resid"]:
        err = (banc[est] - banc.f_out).abs().median()
        print(f"banc, {est} : Spearman(estimation, f_out) = {spearmanr(banc[est], banc.f_out).statistic:.2f} ; "
              f"erreur absolue médiane = {err:.2f}")
    print("\n=== réel : estimation sans étiquettes contre mesure avec étiquettes (vB22b) ===")
    print(t[t.lieu != "banc"][["lieu", "etiquettes", "tendance", "sfa_resid", "r"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
