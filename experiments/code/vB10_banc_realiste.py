"""vB10 — Banc synthétique rendu réaliste : `span_W` tient-il quand les hypothèses idéales de vB09 tombent ?

vB09 : banc idéal (W observé sans bruit, toutes les conditions mesurées, dégradation sans bruit de
procédé, même signature pour toutes les unités, pas de constante d'unité). `span_W` y est robuste
(0,79–0,88) ; la SFA sans W n'est piégée que si la dérive d'usage domine la variation rapide.

Nuisances ajoutées, déclarées avant mesure (niveau « réaliste » / niveau « fort » pour l'ablation) :
  - sW : bruit de mesure sur W observé (écart-type 0,1 / 0,3 ; unités de c) ;
  - sh : une condition NON mesurée (5e dimension de c, rapide, écart-type 0,5 / 1,0), dont l'effet passe
         par le même espace que les autres ;
  - sm : bruit de procédé sur la dégradation, indépendant d'un vol à l'autre (maintenance ; rend d
         localement non monotone ; 0,03 / 0,1, pour une amplitude de vie ≈ 1,3) ;
  - het : hétérogénéité de la signature de dégradation entre unités (A_d,u = A_d + het·perturbation
          renormalisée ; 0,2 / 0,5) ;
  - se : bruit des capteurs (0,1 / 0,3 ; 0,05 dans vB09) ;
  - so : constantes d'unité sur les capteurs (0 / 1,0) — **ablation seulement** : c'est la prop. 3 de la
         voie A, et son remède (SFA centrée par unité) est testé par A (a5, E4).
Toujours λ = 1 (effets non linéaires) ; le reste comme vB09 (6 dimensions d'effets, 3 unités
d'entraînement avec dérive δ, 1 référence, 2 non vues décalées de 4 et 8, 10 graines).

Plan :
  - PRINCIPAL : δ ∈ {0, 3, 10} × σ_f ∈ {1 ; 0,25} × {idéal, réaliste} ;
  - ABLATION : aux deux cellules de dérive forte (δ = 10, σ_f = 1 et 0,25) : idéal + UNE nuisance au
    niveau fort, pour chacune.
Méthodes : `sfa_14` (SFA sans W non réduite, `sfa.ajuster` — le repère de la voie A), `sans_W_v99`,
`insens_W_v99` et `span_W` (orientation à conditions égales, vB08b). Mesure : Spearman(score, d) sur
les non vues (d = dégradation vraie, bruit de procédé compris, comme les étiquettes de N-CMAPSS).

Lecture fixée AVANT de mesurer (Spearman médian, non vues) :
  - réaliste contre idéal : `span_W` perd ≤ 0,1 dans les 6 cellules, et reste au-dessus de `sfa_14` là
    où la dérive domine (σ_f = 0,25, δ ≥ 3) → le résultat de vB09 tient hors des hypothèses idéales ;
  - ablation : on décrit quelle nuisance dégrade le plus `span_W` ; attendu : le bruit sur W (sW) quand la
    variation rapide est faible (il atténue l'estimation de l'effet des conditions), et les constantes
    d'unité (so) pour toutes les SFA (prop. 3 de A).

Machine : Mac, quelques minutes. Sortie : voie_B_avec_exo/results/vB10_*.csv, vB10.txt
"""
from __future__ import annotations

import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402  (ajoute aussi ../code au chemin)
import vB05c_derive_regularisee as d5c  # noqa: E402
import vB08_espace_conditions as v8  # noqa: E402
import vB08b_orientation as v8b  # noqa: E402
import sfa  # noqa: E402

K, KW, RC, N_VOLS, N_GRAINES, LAM = 14, 4, 6, 90, 10, 1.0
S_COLS, W_COLS = v.S_COLS, v.W_COLS
IDEAL = dict(sW=0.0, sh=0.0, sm=0.0, het=0.0, se=0.05, so=0.0)
REALISTE = dict(sW=0.1, sh=0.5, sm=0.03, het=0.2, se=0.1, so=0.0)
FORT = dict(sW=0.3, sh=1.0, sm=0.1, het=0.5, se=0.3, so=1.0)


def psi(c):
    prod = np.stack([c[:, i] * c[:, j] for i, j in combinations(range(c.shape[1]), 2)], 1)
    return np.hstack([c ** 2, prod])


def generer(graine, delta, sigma_f, nz):
    rng = np.random.default_rng(graine)
    kc = KW + 1                                                     # 4 mesurées + 1 non mesurée
    B = np.linalg.qr(rng.normal(size=(K, RC)))[0]
    M1 = rng.normal(size=(RC, kc)) / np.sqrt(kc)
    M2 = rng.normal(size=(RC, kc * (kc + 1) // 2)) / np.sqrt(kc * (kc + 1) // 2)
    e = rng.normal(size=KW); e /= np.linalg.norm(e)
    e2 = rng.normal(size=KW); e2 /= np.linalg.norm(e2)
    dedans = B @ rng.normal(size=RC); dedans /= np.linalg.norm(dedans)
    dehors = (np.eye(K) - B @ B.T) @ rng.normal(size=K); dehors /= np.linalg.norm(dehors)
    A0 = (dedans + dehors) / np.linalg.norm(dedans + dehors)
    roles = {1: ("train", 0.0, delta), 2: ("train", 0.0, delta), 3: ("train", 0.0, delta),
             4: ("ref", 0.0, 0.0), 5: ("non_vue", 4.0, 0.0), 6: ("non_vue", 8.0, 0.0)}
    t = np.linspace(0, 1, N_VOLS)
    parts = []
    for u, (role, decal, dlt) in roles.items():
        c = decal * e2 + dlt * (t[:, None] - 0.5) * e + sigma_f * rng.normal(size=(N_VOLS, KW))
        cache = nz["sh"] * rng.normal(size=(N_VOLS, 1))
        cf = np.hstack([c, cache])
        h = cf @ M1.T + LAM * psi(cf / 2) @ M2.T
        pert = rng.normal(size=K); pert /= np.linalg.norm(pert)
        A_u = A0 + nz["het"] * pert
        A_u = nz.get("a", 0.5) * A_u / np.linalg.norm(A_u)      # amplitude de la signature (vB11)
        d = (rng.uniform(0, 0.1) + 0.3 * t + rng.uniform(0.5, 1.5) * t ** 4
             + nz["sm"] * rng.normal(size=N_VOLS))
        x = (h @ B.T + np.outer(d, A_u) + nz["so"] * rng.normal(size=K)
             + rng.normal(scale=nz["se"], size=(N_VOLS, K)))
        W_obs = c + nz["sW"] * rng.normal(size=c.shape)
        f = pd.DataFrame(x, columns=S_COLS).join(pd.DataFrame(W_obs, columns=W_COLS))
        parts.append(f.assign(unit=u, cycle=np.arange(N_VOLS), role=role, degradation=d))
    return pd.concat(parts, ignore_index=True)


def evaluer_config(delta, sigma_f, nz, etiquette, rows):
    for g in range(N_GRAINES):
        df = generer(g, delta, sigma_f, nz)
        tr = df[df.role == "train"]
        X, W, un, cy = tr[S_COLS].values, tr[W_COLS].values, tr.unit.values, tr.cycle.values
        modeles = {"sfa_14": sfa.ajuster(X, un, cy), "sans_W_v99": sfa.ajuster_reduite(X, un, cy)[0]}
        for nom, m in [("insens_W_v99", d5c.insens_v99(X, W, un, cy)), ("span_W", v8.span_W(X, W, un, cy)[0])]:
            modeles[nom] = sfa.SFA(m.moyenne, m.ecart_type, v8b.signe_partiel(m, tr) * m.w)
        for nom, m in modeles.items():
            for u in (5, 6):
                x = df[df.unit == u]
                rows.append(dict(delta=delta, sigma_f=sigma_f, banc=etiquette, graine=g, methode=nom,
                                 unite=u, spearman=spearmanr(m.score(x[S_COLS].values), x.degradation).statistic))


def main():
    rows = []
    for delta, sigma_f in product([0.0, 3.0, 10.0], [1.0, 0.25]):
        evaluer_config(delta, sigma_f, IDEAL, "idéal", rows)
        evaluer_config(delta, sigma_f, REALISTE, "réaliste", rows)
    for sigma_f in [1.0, 0.25]:
        for cle, val in FORT.items():
            nz = dict(IDEAL); nz[cle] = val
            evaluer_config(10.0, sigma_f, nz, f"idéal + {cle} fort", rows)
    t = pd.DataFrame(rows)
    t.to_csv(v.OUT / "vB10_banc_realiste.csv", index=False)
    ordre = ["sfa_14", "sans_W_v99", "insens_W_v99", "span_W"]
    fmt = lambda s: f"{s.median():.2f} [{s.quantile(0.25):.2f}]"   # noqa: E731
    pd.set_option("display.width", 220)
    print("=== PRINCIPAL — Spearman médian des non vues [quartile bas], λ = 1 ===")
    p = t[t.banc.isin(["idéal", "réaliste"])].groupby(["sigma_f", "delta", "banc", "methode"]).spearman.agg(fmt)
    print(p.unstack("methode")[ordre].to_string())
    print("\n=== ABLATION — δ = 10, une nuisance forte à la fois ===")
    a = t[(t.delta == 10.0) & ~t.banc.isin(["réaliste"])]
    print(a.groupby(["sigma_f", "banc", "methode"]).spearman.agg(fmt).unstack("methode")[ordre].to_string())


if __name__ == "__main__":
    main()
