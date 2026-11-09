"""vB26 — Baselines de la littérature sur nos trois jeux avec W (papier B).

Baselines, déclarées avant mesure (réseaux 2 × 64, ReLU, Adam lr 1e-3, lots de 512, 60 epochs, 1 thread,
entrées standardisées sur l'entraînement ; AUCUN réglage) :
  - `residu_FNN` : résidu standard (baseline de H-CDE, réf. [18] de leur papier ; très répandu) : un
    réseau apprend W → S sur le « sain » supposé (20 % premiers de la vie de chaque unité d'entraînement,
    aucune étiquette) ; indicateur = norme du résidu ‖S − f(W)‖ ;
  - `bajarunas_corr`, `bajarunas_mono` : RÉIMPLÉMENTATION d'après Bajarunas et al. 2024 (§4, architecture
    causale W → Z → X) — pas leur code : encodeur S → z (scalaire), décodeur (W, z) → S ; perte =
    reconstruction + λ·contrainte, λ = 1 ; `corr` : −corrélation (Pearson) entre z et le rang du vol dans la
    vie, par unité dans le lot (leur contrainte de trendabilité) ; `mono` : pénalité relu(z_i − z_j) sur les
    couples de lignes d'une même unité avec t_j > t_i (version par couples de leur « gradient négatif ») ;
    leur contrainte « fonctionnelle » (courbe de fiabilité propre au système) n'est pas reprise.
Cas et mesures : les 20 cas de vB24 sans dérive fabriquée — N-CMAPSS DS01/05/07 (classe non vue ;
Spearman par vol avec la dégradation, non vues), C-MAPSS FD002/FD004 × 6 cas de a6 (Spearman avec le cycle
sur le régime cible ; graine = tirage), batteries × 4 groupes (Spearman avec la perte de capacité par
cycle). 5 graines par cas (C-MAPSS : les 5 tirages). Comparaison : la méthode choisie par la règle v1
(vB24) sur le même cas.

Lecture fixée AVANT de mesurer :
  - pour chaque cas, écart = règle − meilleure baseline (médiane des graines) ; le papier B peut dire que
    la règle égale ou dépasse les baselines si la médiane des écarts sur les 20 cas est ≥ −0,02 ;
  - on rapporte les cas où une baseline gagne de plus de 0,1, sans les expliquer après coup.
Réserve : réimplémentation (pas le code des auteurs), sans leur réglage ; résultats de Bajarunas à citer
comme « d'après leur méthode ».

Machine : Mac, processus parallèles mono-thread. Sortie : voie_B_avec_exo/results/vB26_*.csv, vB26.txt
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))

EPOCHS, LOT, LAMBDA, SAIN = 60, 512, 1.0, 0.2
BASELINES = ["residu_FNN", "bajarunas_corr", "bajarunas_mono"]


def mlp(n_in, n_out):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_out))


def entrainer(S, W, unit, rang, graine):
    """Renvoie les trois fonctions de score (S, W) → indicateur, apprises sur l'entraînement."""
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    mS, sS, mW, sW = S.mean(0), S.std(0) + 1e-12, W.mean(0), W.std(0) + 1e-12
    Sz = torch.tensor((S - mS) / sS, dtype=torch.float32)
    Wz = torch.tensor((W - mW) / sW, dtype=torch.float32)
    u = torch.tensor(pd.factorize(unit)[0])
    t = torch.tensor(rang, dtype=torch.float32)
    n, k, kw = len(S), S.shape[1], W.shape[1]
    out = {}
    # résidu standard, sur le début de vie
    sain = np.where(rang < SAIN)[0]
    f = mlp(kw, k)
    opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        for i in np.array_split(rng.permutation(sain), max(1, len(sain) // LOT)):
            opt.zero_grad()
            ((f(Wz[i]) - Sz[i]) ** 2).mean().backward()
            opt.step()
    f.eval()
    out["residu_FNN"] = lambda S_, W_, f=f: torch.linalg.norm(
        torch.tensor((S_ - mS) / sS, dtype=torch.float32) - f(torch.tensor((W_ - mW) / sW, dtype=torch.float32)),
        dim=1).detach().numpy()
    # réimplémentation Bajarunas : encodeur S → z, décodeur (W, z) → S
    for nom in ["bajarunas_corr", "bajarunas_mono"]:
        enc, dec = mlp(k, 1), mlp(kw + 1, k)
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
        for _ in range(EPOCHS):
            for i in np.array_split(rng.permutation(n), max(1, n // LOT)):
                i = torch.tensor(i)
                z = enc(Sz[i])
                perte = ((dec(torch.cat([Wz[i], z], 1)) - Sz[i]) ** 2).mean()
                c = []
                for un in torch.unique(u[i]):
                    m = u[i] == un
                    if m.sum() < 10:
                        continue
                    zu, tu = z[m, 0], t[i][m]
                    if nom == "bajarunas_corr":
                        zc, tc = zu - zu.mean(), tu - tu.mean()
                        c.append(-(zc * tc).sum() / (zc.norm() * tc.norm() + 1e-8))
                    else:
                        avant = tu[:, None] < tu[None, :]
                        c.append(torch.relu(zu[:, None] - zu[None, :])[avant].mean())
                if c:
                    perte = perte + LAMBDA * torch.stack(c).mean()
                opt.zero_grad()
                perte.backward()
                opt.step()
        enc.eval()
        out[nom] = lambda S_, W_, enc=enc: enc(torch.tensor((S_ - mS) / sS, dtype=torch.float32))[:, 0].detach().numpy()
    return out


# --------------------------------------------------------------------------------------------------
def tache(args):
    jeu, cas, graine = args
    import vB01_sonde_marge as v
    rows = []
    if jeu == "N-CMAPSS":
        import vB02b_support_et_modes as b
        from evaluateur import _joindre, etiquettes, par_vol
        d = b.charger(cas)
        tr = pd.concat([d["f"][u] for u in d["train"]], ignore_index=True)
        rang = tr.groupby("unit").cycle.rank(pct=True).values
        fs = entrainer(tr[v.S_COLS].values, tr[v.W_COLS].values, tr.unit.values, rang, graine)
        for nom, f in fs.items():
            for u in d["unseen"]:
                x = d["f"][u]
                e = par_vol(_joindre(x[["unit", "cycle"]].assign(score=f(x[v.S_COLS].values, x[v.W_COLS].values)),
                                     etiquettes(cas)))
                rows.append(dict(jeu=jeu, cas=f"{cas} aucune", graine=graine, methode=nom, unite=u,
                                 perf=spearmanr(e.score, e.degradation).statistic))
    elif jeu == "C-MAPSS":
        import vB19_cmapss_avec_w as b19
        fd, nom_cas = cas.split()
        c = [x for x in b19.a6.CAS if x[0] == nom_cas][0]
        _, retire, cible = c
        dd = b19.a6.charger(fd)
        rng = np.random.default_rng(graine)
        uu = dd.unit.unique()
        tr_u = set(rng.choice(uu, size=len(uu) // 2, replace=False))
        tr = dd[dd.unit.isin(tr_u)]
        if retire is not None:
            tr = tr[tr.regime != retire]
        te = dd[~dd.unit.isin(tr_u)]
        X = [x for x in b19.a6.COLS[5:] if tr[x].std() > 1e-8]
        Wk = [x for x in b19.W_C if tr[x].std() > 1e-8]
        rang = tr.cycle.values / tr.groupby("unit").cycle.transform("max").values
        fs = entrainer(tr[X].values.astype(float), tr[Wk].values.astype(float), tr.unit.values, rang, graine)
        for nom, f in fs.items():
            s = f(te[X].values.astype(float), te[Wk].values.astype(float))
            t = te.assign(s=s)
            sp = [spearmanr(g[g.regime == cible].cycle, g[g.regime == cible].s).statistic
                  for _, g in t.groupby("unit") if (g.regime == cible).sum() >= 10]
            rows.append(dict(jeu=jeu, cas=cas, graine=graine, methode=nom, unite=-1, perf=float(np.median(sp))))
    else:
        import vB22_batteries as b22
        df = b22.charger()
        grp = {b_: g for g, bs in b22.GROUPES.items() for b_ in bs}
        df["groupe"] = df.bat.map(grp)
        tr = df[df.groupe == "uniforme"]
        rang = tr.cycle.values / tr.groupby("bat").cycle.transform("max").values
        fs = entrainer(tr[b22.S_B].values, tr[b22.W_B].values, tr.bat.values, rang, graine)
        te = df[df.groupe != "uniforme"]
        for nom, f in fs.items():
            t = te.assign(s=f(te[b22.S_B].values, te[b22.W_B].values))
            fv = t.groupby(["groupe", "bat", "cycle"]).agg(s=("s", "mean"), cap=("capacite", "first")).reset_index()
            for (g, b_), x in fv.groupby(["groupe", "bat"]):
                rows.append(dict(jeu=jeu, cas=g, graine=graine, methode=nom, unite=b_,
                                 perf=spearmanr(x.s, -x.cap).statistic))
    return rows


def main():
    import vB01_sonde_marge as v
    taches = [("N-CMAPSS", ds, g) for ds in ["DS01", "DS05", "DS07"] for g in range(5)]
    taches += [("C-MAPSS", f"{fd} {c}", g) for fd in ["FD002", "FD004"]
               for c in ["C0_42", "C0_60", "C0_20", "R42", "R60", "R20"] for g in range(5)]
    taches += [("batteries", "tous", g) for g in range(5)]
    with Pool(10) as pool:
        res = pool.map(tache, taches, chunksize=1)
    t = pd.DataFrame([r for rr in res for r in rr])
    t.to_csv(v.OUT / "vB26_baselines.csv", index=False)
    # performance par cas : médiane sur les unités puis sur les graines
    p = t.groupby(["jeu", "cas", "graine", "methode"]).perf.median().groupby(["jeu", "cas", "methode"]).median()
    p = p.unstack("methode")[BASELINES].reset_index()
    r24 = pd.read_csv(v.OUT / "vB24_regle_choix.csv")
    r24 = r24[~r24.cas.str.contains("montante")]
    r24["regle"] = r24.apply(lambda r: r[r.choix], axis=1)
    m = r24[["jeu", "cas", "choix", "regle", "lenteur", "modele", "span_W"]].merge(p, on=["jeu", "cas"])
    m["meilleure_baseline"] = m[BASELINES].max(axis=1)
    m["ecart"] = m.regle - m.meilleure_baseline
    m.to_csv(v.OUT / "vB26_comparaison.csv", index=False)
    pd.set_option("display.width", 250)
    print(m.round(2).to_string(index=False))
    print(f"\nmédiane des écarts (règle − meilleure baseline) sur {len(m)} cas : {m.ecart.median():.3f}")
    print("cas où une baseline gagne de plus de 0,1 :")
    print(m[m.ecart < -0.1][["jeu", "cas", "regle", "meilleure_baseline"]].round(2).to_string(index=False) or "aucun")


if __name__ == "__main__":
    main()
