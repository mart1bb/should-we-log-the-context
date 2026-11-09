"""vB45 — Labels de régime consommés par une minimisation du risque invariant (V-REx), R16 de PAPIER_ICDE § 8.

La relecture (W2) note qu'IRM est citée mais pas évaluée. IRM suppose une perte supervisée ; on prend sa variante
V-REx (Krueger et al. 2021 : pénaliser la variance des risques entre environnements), qui s'écrit pour toute perte,
ici l'objectif de lenteur de vB34, avec les régimes comme environnements :
  - encodeur de vB34 (MLP k → 64 → 64 → 1), mêmes lots, Adam 1e-3, 3 000 pas ;
  - risque de l'environnement e : L_e = moyenne des (z_{t+1} − z_t)² sur les couples consécutifs d'une même unité dont
    la première fenêtre est dans le régime e, divisée par la variance de z sur un lot de lignes au hasard ;
  - perte = moyenne_e L_e + β · variance_e(L_e) ; β ∈ {1, 10, 100} (β = 10 = réglage principal, déclaré) ; 3 graines ;
    signe : z croît avec le rang du vol sur l'entraînement ; régimes de vB33 (natifs sur C-MAPSS, k-moyennes K = 6
    ailleurs). Mêmes cas, données et mesure que vB39 (tâches de vB39).
Lecture fixée AVANT de mesurer (par cas : écart à la meilleure sans contexte de vB39 ; par jeu : médiane ; jeux
informatifs ; C-MAPSS : régimes retenus) : « les labels servent via V-REx » sur un jeu si l'écart médian ≥ 0,05 avec
β = 10 ; rapporté aussi : meilleur β choisi avec la vérité (oracle), même critère. Attendu (déclaré) : non partout.

Machine : Mac (choix de Martin pour la révision). Sortie : results/vB45_par_unite.csv, vB45.txt
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB39_banc_par_unite as b39  # noqa: E402
import vB42_quoi_journaliser as b42  # noqa: E402

BETAS, GRAINES, PAS, LOT = (1.0, 10.0, 100.0), (0, 1, 2), 3000, 512


def entrainer(Sz, idx_a, idx_b, lab, beta, graine, rang):
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    k = Sz.shape[1]
    enc = nn.Sequential(nn.Linear(k, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    X = torch.tensor(Sz, dtype=torch.float32)
    env = lab[idx_a]
    envs = [e for e in np.unique(env) if (env == e).sum() >= 10]
    par_env = {e: np.where(env == e)[0] for e in envs}
    for _ in range(PAS):
        r = torch.tensor(rng.integers(0, len(Sz), LOT))
        var = enc(X[r]).var() + 1e-6
        risques = []
        for e in envs:
            c = par_env[e][rng.integers(0, len(par_env[e]), max(8, LOT // len(envs)))]
            risques.append(((enc(X[idx_b[c]]) - enc(X[idx_a[c]])) ** 2).mean() / var)
        R = torch.stack(risques)
        perte = R.mean() + (beta * R.var() if len(risques) > 1 else 0.0)
        opt.zero_grad()
        perte.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        z = enc(X)[:, 0].numpy()
    signe = 1.0 if np.corrcoef(z, rang)[0, 1] >= 0 else -1.0
    return lambda Sz_: signe * enc(torch.tensor(Sz_, dtype=torch.float32))[:, 0].detach().numpy()


def vrex(S, W, unit, cycle, labels=None):
    import vB33_labels_regime as m33
    from sklearn.cluster import KMeans
    unit, cycle = np.asarray(unit), np.asarray(cycle)
    mu, sd = S.mean(0), S.std(0) + 1e-12
    Sz = (S - mu) / sd
    Wz = (W - W.mean(0)) / (W.std(0) + 1e-12)
    lab = (pd.factorize(np.asarray(labels))[0] if labels is not None
           else KMeans(m33.K, n_init=10, random_state=0).fit_predict(Wz))
    ordre = np.lexsort((np.arange(len(S)), cycle, unit))
    u_o = unit[ordre]
    meme = u_o[1:] == u_o[:-1]
    idx_a, idx_b = ordre[:-1][meme], ordre[1:][meme]
    rang = b39._rang(unit, cycle)
    out = {}
    for beta in BETAS:
        for g in GRAINES:
            f = entrainer(Sz, idx_a, idx_b, lab, beta, g, rang)
            out[f"vrex_b{int(beta)}_g{g}"] = (lambda S_, W_, f=f: f((S_ - mu) / sd))
    return out


def lancer(t):
    b39.tout = vrex
    return b39.lancer(t)


def main():
    taches = [t for t in b42.taches_base() if t[0] != "P"]
    seul = os.environ.get("VB45_SEUL")
    if seul:
        taches = [t for t in taches if t[0] == seul][:1]
    taches = sorted(taches, key=lambda x: 0 if x[0] in ("N", "R") else 1)
    with Pool(int(os.environ.get("VB45_PROC", "10"))) as pool:
        L = pd.DataFrame([r for rr in pool.map(lancer, taches, chunksize=1) for r in rr])
    suf = f"_essai_{seul}" if seul else ""
    L.to_csv(v.OUT / f"vB45_par_unite{suf}.csv", index=False)
    C = b39.par_cas(L)
    C.loc[C.jeu == "C-MAPSS", "jeu"] = "C-MAPSS retenu"
    cf = b42.sans_contexte()
    C = C.merge(cf, on=["jeu", "cas"])
    meths = [f"vrex_b{int(b)}" for b in BETAS]
    ordre = ["N-CMAPSS", "C-MAPSS retenu", "batteries", "roulements", "gaz"]
    med = C.groupby("jeu")[meths + ["cf"]].median().reindex(ordre)
    gain = C[meths].sub(C.cf, axis=0).groupby(C.jeu).median().reindex(ordre)
    with open(v.OUT / f"vB45{suf}.txt", "w") as fh:
        def p(*a):
            print(*a)
            print(*a, file=fh)
        p("=== V-REx (lenteur, régimes = environnements) : médiane par jeu ===")
        p(med.round(2).to_string())
        p("\n=== gain médian apparié sur la meilleure sans contexte ===")
        p(gain.round(3).to_string())
        for j in ordre:
            if j not in gain.index or gain.loc[j].isna().all():
                continue
            p(f"  {j:16s} β = 10 : {gain.loc[j, 'vrex_b10']:+.3f} → "
              f"{'les labels servent' if gain.loc[j, 'vrex_b10'] >= 0.05 else 'non'} ; oracle β : "
              f"{gain.loc[j].max():+.3f} → {'servent' if gain.loc[j].max() >= 0.05 else 'non'}")


if __name__ == "__main__":
    main()
