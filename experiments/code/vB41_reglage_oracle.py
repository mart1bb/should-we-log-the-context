"""vB41 — Résidu FNN et DANN réglés en ORACLE (R11 de PAPIER_ICDE § 8) : « mal réglé » explique-t-il leur échec ?

Objection de la relecture (W2) : le FNN et DANN ne sont pas réglés. Sans étiquettes, il n'existe pas de critère de
réglage sur des contextes jamais vus ; on donne donc à ces méthodes un avantage qu'aucun praticien n'aurait : une
grille de réglages, et le meilleur réglage choisi AVEC la vérité terrain du jeu. C'est une borne haute.
Grilles déclarées (une graine, 0 ; mêmes cas, données et mesure que vB39 ; tâches de vB39) :
  - FNN résidu (vB26) : largeur {32, 128} × profondeur {2, 3} × epochs {60, 200} × part « saine » du début de vie
    {0,2 ; 0,5} = 16 réglages ; Adam 1e-3, lots de 512, ≤ 50 000 lignes (comme vB35) ;
  - DANN (vB34, `dann_lent`) : λ {0,03 ; 0,1 ; 0,3 ; 1} × pas {1 000 ; 3 000 ; 10 000} = 12 réglages ; architecture
    et lots de vB34.
Deux oracles par jeu : (a) réglage unique, de meilleure médiane sur les cas du jeu ; (b) meilleur réglage PAR CAS
(borne encore plus haute).

Lecture fixée AVANT de mesurer (jeux informatifs ; C-MAPSS : régimes retenus) :
  - « le réglage explique l'échec » sur un jeu si l'oracle (a) ≥ meilleure sans contexte (médiane, vB39) + 0,05 ;
  - sinon : « même réglée avec les étiquettes, la méthode ne bat pas l'a priori temporel sans contexte » sur ce jeu.
Attendu (déclaré) : la seconde lecture partout sauf peut-être les batteries pour le FNN.

Machine : choisie par Martin. Sortie : results/vB41_par_unite.csv, vB41.txt
"""
from __future__ import annotations

import itertools
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

GRILLE_FNN = list(itertools.product([32, 128], [2, 3], [60, 200], [0.2, 0.5]))
GRILLE_DANN = list(itertools.product([0.03, 0.1, 0.3, 1.0], [1000, 3000, 10000]))


def nom_fnn(c):
    return f"FNN_w{c[0]}_d{c[1]}_e{c[2]}_s{c[3]}"


def nom_dann(c):
    return f"DANN_l{c[0]}_p{c[1]}"


def fnn(S, W, unit, cycle, largeur, prof, epochs, part):
    import torch
    nn = torch.nn
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    rang = b39._rang(unit, cycle)
    sel = (np.sort(np.random.default_rng(0).choice(len(S), b39.t35.MAX_NN, replace=False))
           if len(S) > b39.t35.MAX_NN else np.arange(len(S)))
    S, W, rang = S[sel], W[sel], rang[sel]
    mS, sS, mW, sW = S.mean(0), S.std(0) + 1e-12, W.mean(0), W.std(0) + 1e-12
    Sz = torch.tensor((S - mS) / sS, dtype=torch.float32)
    Wz = torch.tensor((W - mW) / sW, dtype=torch.float32)
    couches, d = [], W.shape[1]
    for _ in range(prof):
        couches += [nn.Linear(d, largeur), nn.ReLU()]
        d = largeur
    f = nn.Sequential(*couches, nn.Linear(d, S.shape[1]))
    opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    sain = np.where(rang < part)[0]
    for _ in range(epochs):
        for i in np.array_split(rng.permutation(sain), max(1, len(sain) // 512)):
            opt.zero_grad()
            ((f(Wz[i]) - Sz[i]) ** 2).mean().backward()
            opt.step()
    f.eval()
    return lambda S_, W_: torch.linalg.norm(
        torch.tensor((S_ - mS) / sS, dtype=torch.float32) - f(torch.tensor((W_ - mW) / sW, dtype=torch.float32)),
        dim=1).detach().numpy()


def grille(S, W, unit, cycle, labels=None):
    import vB34_dann_regime as d34
    import vB33_labels_regime as m33
    from sklearn.cluster import KMeans
    unit, cycle = np.asarray(unit), np.asarray(cycle)
    out = {nom_fnn(c): fnn(S, W, unit, cycle, *c) for c in GRILLE_FNN}
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
    for lam, pas in GRILLE_DANN:
        d34.LAMBDA, d34.PAS = lam, pas                      # lus par d34.entrainer à l'appel
        f = d34.entrainer(Sz, idx_a, idx_b, lab, True, 0, rang)
        out[nom_dann((lam, pas))] = (lambda S_, W_, f=f: f((S_ - mu) / sd))
    return out


def lancer(t):
    b39.tout = grille                                       # les tâches de vB39 appellent b39.tout
    return b39.lancer(t)


def main():
    import vB05_derive_usage as d5
    import vB19_cmapss_avec_w as b19
    import vB22_batteries as b22
    import vB29_roulements_upb as u29
    import vB31_pvdaq as p31
    taches = []
    for ds in d5.DS_:
        fc = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv", usecols=["unit", "Fc"]).groupby("unit").Fc.first()
        taches += [("N", (ds, ref)) for ref in sorted(fc[fc == 1].index)]
    taches += [("C", (fd, cas, ret, cib, t)) for fd in ["FD002", "FD004"] for cas, ret, cib in b19.a6.CAS
               for t in range(b19.N_TIRAGES) if ret is not None]          # régimes retenus seulement
    taches += [("B", g) for g in b22.GROUPES] + [("R", g) for g in u29.GROUPES]
    taches += [("G", (gz, imp)) for gz in range(1, 7) for imp in (True, False)]
    seul = os.environ.get("VB41_SEUL")
    if seul:
        taches = [t for t in taches if t[0] == seul][:1]
    taches = [t for t in taches if t[0] in ("R", "N")] + [t for t in taches if t[0] not in ("R", "N")]
    with Pool(int(os.environ.get("VB41_PROC", "10"))) as pool:
        L = pd.DataFrame([r for rr in pool.map(lancer, taches, chunksize=1) for r in rr])
    suffixe = f"_essai_{seul}" if seul else ""
    L.to_csv(v.OUT / f"vB41_par_unite{suffixe}.csv", index=False)
    C = L.groupby(["jeu", "cas", "methode"]).rho.median().unstack("methode").reset_index()
    cm = C.jeu == "C-MAPSS"
    C.loc[cm, "jeu"] = "C-MAPSS retenu"
    ordre = ["N-CMAPSS", "C-MAPSS retenu", "batteries", "roulements", "gaz"]
    ref = pd.read_csv(v.OUT / "vB39_cas.csv") if (v.OUT / "vB39_cas.csv").exists() else None
    pd.set_option("display.width", 250)
    with open(v.OUT / f"vB41{suffixe}.txt", "w") as fh:
        def p(*a):
            print(*a)
            print(*a, file=fh)
        for fam, noms in [("FNN", [nom_fnn(c) for c in GRILLE_FNN]), ("DANN", [nom_dann(c) for c in GRILLE_DANN])]:
            med = C.groupby("jeu")[noms].median().reindex(ordre)
            p(f"\n=== {fam} : médiane par jeu, chaque réglage ===")
            p(med.T.round(2).to_string())
            oa = med.max(axis=1)
            ob = C.groupby("jeu")[noms].apply(lambda d: d.max(axis=1).median()).reindex(ordre)
            p(f"\n{fam} oracle (a) réglage unique par jeu : " + oa.round(2).to_string().replace("\n", " ; "))
            p(f"{fam} oracle (b) meilleur réglage par cas : " + ob.round(2).to_string().replace("\n", " ; "))
            if ref is not None:
                r = ref.copy()
                r.loc[(r.jeu == "C-MAPSS") & r.cas.str.contains(r" R\d"), "jeu"] = "C-MAPSS retenu"
                cf = r.assign(cf=r[["lenteur", "tendance_moy", "tendance_reg"]].max(axis=1)).groupby("jeu").cf.median()
                for j in ordre:
                    verdict = "le réglage explique l'échec" if oa[j] >= cf[j] + 0.05 else "ne bat pas le sans-contexte"
                    p(f"  {j:16s} oracle {oa[j]:.2f} / meilleure sans contexte {cf[j]:.2f} → {verdict}")


if __name__ == "__main__":
    main()
