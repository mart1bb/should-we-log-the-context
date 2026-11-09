"""vB34 — Labels de régime consommés par un apprentissage adversarial de domaine NON LINÉAIRE (DANN) : servent-ils ?

Suite de vB33 (version linéaire : les labels ne servent sur aucun jeu). Limite levée ici : DANN non linéaire
(Ganin et al. 2016), non supervisé (aucune étiquette de santé), avec le même a priori temporel que la lenteur :
  - `lent_nl` (contrôle) : encodeur MLP S → z (k → 64 → 64 → 1, ReLU), entraîné à rendre z lent : perte =
    moyenne des (z_{t+1} − z_t)² sur des couples de lignes consécutives d'une même unité, divisée par la variance
    de z sur un lot de lignes tirées au hasard (évite l'effondrement) ;
  - `dann_lent` : même encodeur et même perte, plus un adversaire (z → 32 → nombre de régimes) qui cherche à
    reconnaître le régime depuis z, branché par inversion de gradient (λ = 1) : z doit être lent ET ne pas porter
    le régime.
Réglages déclarés, aucun réglage après mesure : capteurs standardisés sur l'entraînement ; couples consécutifs au
niveau des lignes (fenêtre, cycle, étape, mesure), unités jamais mélangées ; Adam lr 1e-3 ; 3 000 pas ; lots de
512 couples et 512 lignes ; 3 graines (performance = médiane des graines) ; 1 thread par entraînement (D-17) ;
signe : z croît avec le rang du vol sur l'entraînement. Régimes : ceux de vB33 (natifs sur C-MAPSS, k-moyennes
K = 6 sur W ailleurs). Jeux, découpages, vérités : ceux de vB32/vB33 (mêmes intitulés de cas), par les fonctions de
tâche de vB33.

Lecture fixée AVANT de mesurer : par cas, gain = dann_lent − meilleure méthode sans contexte (max de lenteur,
tendance_moy, tendance_reg, relues dans vB32) ; par jeu, médiane des gains. « Les labels servent (DANN) » sur un jeu si
la médiane ≥ 0,05. Rapporté aussi : effet propre des labels = dann_lent − lent_nl (même architecture, sans
adversaire). PVDAQ rapporté à part (non informatif).

Machine : Mac (Martin, 12 oct.), processus parallèles mono-thread. Sortie : voie_B_avec_exo/results/vB34_*.csv, vB34.txt
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB33_labels_regime as m33  # noqa: E402

PAS, LOT, LAMBDA, GRAINES = 3000, 512, 1.0, (0, 1, 2)
METH = ["lent_nl", "dann_lent"]


def entrainer(Sz, idx_a, idx_b, lab, adversaire, graine, rang):
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    k, n_reg = Sz.shape[1], int(lab.max()) + 1
    enc = nn.Sequential(nn.Linear(k, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    adv = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, n_reg))
    params = list(enc.parameters()) + (list(adv.parameters()) if adversaire else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    X = torch.tensor(Sz, dtype=torch.float32)
    L = torch.tensor(lab, dtype=torch.long)

    class Inverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -LAMBDA * g

    for _ in range(PAS):
        c = rng.integers(0, len(idx_a), LOT)
        r = torch.tensor(rng.integers(0, len(Sz), LOT))
        za, zb = enc(X[idx_a[c]]), enc(X[idx_b[c]])
        zr = enc(X[r])
        perte = ((zb - za) ** 2).mean() / (zr.var() + 1e-6)
        if adversaire:
            perte = perte + nn.functional.cross_entropy(adv(Inverse.apply(zr)), L[r])
        opt.zero_grad()
        perte.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        z = enc(X)[:, 0].numpy()
    signe = 1.0 if np.corrcoef(z, rang)[0, 1] >= 0 else -1.0
    return lambda Sz_: signe * enc(torch.tensor(Sz_, dtype=torch.float32))[:, 0].detach().numpy()


def methodes(S, W, unit, cycle, labels=None):
    """Mêmes signature et clés de sortie que vB33 (une clé par méthode et par graine)."""
    mu, sd = S.mean(0), S.std(0) + 1e-12
    Sz = (S - mu) / sd
    Wz = (W - W.mean(0)) / (W.std(0) + 1e-12)
    lab = (pd.factorize(np.asarray(labels))[0] if labels is not None
           else KMeans(m33.K, n_init=10, random_state=0).fit_predict(Wz))
    ordre = np.lexsort((np.arange(len(S)), cycle, unit))
    u_o = np.asarray(unit)[ordre]
    meme = u_o[1:] == u_o[:-1]
    idx_a, idx_b = ordre[:-1][meme], ordre[1:][meme]
    rang = pd.Series(np.asarray(cycle)).groupby(np.asarray(unit)).rank(pct=True).values
    out = {}
    for nom, adversaire in [("lent_nl", False), ("dann_lent", True)]:
        for g in GRAINES:
            f = entrainer(Sz, idx_a, idx_b, lab, adversaire, g, rang)
            out[f"{nom}_g{g}"] = (lambda S_, W_, f=f: f((S_ - mu) / sd))
    return out


def _avec(fn, args=None):
    m33.methodes = methodes                       # les tâches de vB33 appellent m33.methodes
    return fn(args) if args is not None else fn()


def w_ncmapss(a):
    return _avec(m33.t_ncmapss, a)


def w_cmapss(a):
    return _avec(m33.t_cmapss, a)


def w_batteries(a):
    return _avec(m33.t_batteries, a)


def w_simple(nom):
    return _avec({"roulements": m33.t_roulements, "gaz": m33.t_gaz, "PVDAQ": m33.t_pvdaq}[nom])


def main():
    import vB05_derive_usage as d5
    import vB19_cmapss_avec_w as b19
    import vB22_batteries as b22
    tn = []
    for ds in d5.DS_:
        fc = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv", usecols=["unit", "Fc"]).groupby("unit").Fc.first()
        tn += [(ds, sc, ref) for sc in ["aucune", "montante_h0.30"] for ref in sorted(fc[fc == 1].index)]
    tc = [(fd, cas, retire, cible, t) for fd in ["FD002", "FD004"] for cas, retire, cible in b19.a6.CAS
          for t in range(b19.N_TIRAGES)]
    with Pool(10) as pool:
        rs = pool.map_async(w_simple, ["PVDAQ", "roulements", "gaz"], chunksize=1)
        rb = pool.map_async(w_batteries, list(b22.GROUPES), chunksize=1)
        rn = pool.map_async(w_ncmapss, tn, chunksize=1)
        rc = pool.map_async(w_cmapss, tc, chunksize=1)
        rows = [r for rr in rn.get() + rc.get() + rb.get() + rs.get() for r in rr]
    t = pd.DataFrame(rows)
    for m in METH:
        t[m] = t[[f"{m}_g{g}" for g in GRAINES]].median(axis=1)
    t32 = pd.read_csv(v.OUT / "vB32_tendance_contre_lenteur.csv")
    t = t.merge(t32[["jeu", "cas", "lenteur", "tendance_moy", "tendance_reg"]], on=["jeu", "cas"], how="left")
    t["sans_contexte"] = t[["lenteur", "tendance_moy", "tendance_reg"]].max(axis=1)
    t["gain"] = t.dann_lent - t.sans_contexte
    t["effet_labels"] = t.dann_lent - t.lent_nl
    t.to_csv(v.OUT / "vB34_dann_regime.csv", index=False)

    pd.set_option("display.width", 250)
    ordre = ["N-CMAPSS", "C-MAPSS", "batteries", "roulements", "gaz", "PVDAQ"]
    cols = ["sans_contexte", "lent_nl", "dann_lent", "gain", "effet_labels"]
    g = t.groupby("jeu")[cols].median().reindex(ordre)
    g["cas"] = t.groupby("jeu").size().reindex(ordre)
    g["dispersion_graines_dann"] = t.groupby("jeu").apply(
        lambda x: float(np.median(x[[f"dann_lent_g{s}" for s in GRAINES]].std(axis=1)))).reindex(ordre)
    print("=== médianes par jeu ===")
    print(g.round(3).to_string())
    servent = g.drop(index="PVDAQ").gain >= 0.05
    print(f"\njeux (hors PVDAQ) où les labels servent via DANN (gain médian ≥ 0,05) : "
          f"{list(servent[servent].index) or 'aucun'}")


if __name__ == "__main__":
    main()
