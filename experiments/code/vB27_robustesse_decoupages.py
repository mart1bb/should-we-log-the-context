"""vB27 — Robustesse aux découpages : la règle de choix v1 (vB24) tient-elle quand on change les unités ?

vB24 a mesuré la règle sur UN découpage par jeu (N-CMAPSS : plan de la campagne ; batteries : entraînement
sur les uniformes ; C-MAPSS : médiane de 5 tirages). Ici, les indicateurs sans étiquettes (p_hors, D) et
les trois méthodes (`lenteur`, `modele`, `span_W` v2 — mêmes définitions que vB24) sont recalculés pour
chaque découpage, et la règle v1 (inchangée : p_hors < 0,3 → modele ; D > 0,2 → span_W ; sinon lenteur)
est appliquée découpage par découpage.

Découpages, déclarés avant mesure :
  - N-CMAPSS DS01, DS05, DS07 × {sans dérive, dérive d'usage modérée h = 0,30} : les 4 unités Fc1 tournent —
    chacune est à son tour la référence, les 3 autres l'entraînement ; non vues = TOUTES les unités Fc2/Fc3
    (6, au lieu de 2 dans le plan) ; 4 découpages par cas, 24 en tout ;
  - C-MAPSS FD002, FD004 × 6 cas de a6 : les 5 tirages de vB19/vB21, pris un par un au lieu de leur
    médiane (performances relues dans vB19_cmapss.csv et vB21_d_cmapss.csv ; indicateurs recalculés par
    tirage) ; 60 découpages ;
  - batteries : chaque groupe d'usage est à son tour l'entraînement (uniformes, biaisées basses/hautes,
    ambiante/40 °C), test = les 4 autres groupes ; 5 découpages × 4 groupes de test = 20 cas.
Performance : comme vB24 (Spearman médian sur les unités de test ; C-MAPSS : sur le régime cible ;
batteries : contre la perte de capacité, par cycle). XJTU-SY exclu (p_hors non défini, comme vB24).

Lecture fixée AVANT de mesurer (sur l'ensemble des découpages) :
  - stabilité : pour chaque jeu, part des découpages où la règle choisit la même méthode que dans vB24
    pour le même cas (batteries : `modele`) — robuste si ≥ 90 % ;
  - regret de la règle : médiane ≤ 0,02 ; part des découpages à regret > 0,2 inférieure à celle de
    chacune des trois stratégies fixes ;
  - les découpages à regret > 0,2 sont listés ; seuls ceux de C-MAPSS FD002 R20 sont un échec déjà connu,
    les autres sont rapportés sans explication après coup ;
  - dispersion de p_hors par jeu (min–max) et nombre de découpages de chaque côté du seuil 0,3.

Machine : Mac (Martin, 1er oct.), processus parallèles mono-thread. Sortie : voie_B_avec_exo/results/vB27_*.csv,
vB27.txt
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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB05_derive_usage as d5  # noqa: E402
import vB08_espace_conditions as v8  # noqa: E402
import vB19_cmapss_avec_w as b19  # noqa: E402
import vB22_batteries as b22  # noqa: E402
import vB24_regle_choix as r24  # noqa: E402
import sfa  # noqa: E402
from evaluateur import evaluer  # noqa: E402

M = ["lenteur", "modele", "span_W"]


def methodes(S, W, un, cy):
    """Les trois méthodes de vB24, apprises sur l'entraînement ; renvoie des fonctions (S, W) → score."""
    muW, sdW = W.mean(0), W.std(0)
    q = LinearRegression().fit(r24.phi((W - muW) / sdW), S)
    m_len = sfa.ajuster(S, un, cy)
    m_mod = sfa.ajuster(S - q.predict(r24.phi((W - muW) / sdW)), un, cy)
    m_sw = v8.span_W(S, W, un, cy, reduire=False)[0]
    return {"lenteur": lambda S_, W_: m_len.score(S_),
            "modele": lambda S_, W_: m_mod.score(S_ - q.predict(r24.phi((W_ - muW) / sdW))),
            "span_W": lambda S_, W_: m_sw.score(S_)}


# --------------------------------------------------------------------------------------------------
def tache_ncmapss(args):
    ds, nom_sc, ref = args
    v.it10.DILATION = v.parse_key(ds)[1]
    df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
    fc = df.groupby("unit").Fc.first()
    fc1 = sorted(fc[fc == 1].index)
    train = [u for u in fc1 if u != ref]
    unseen = sorted(fc[fc != 1].index)
    complets = {u: v.fenetres(df, u) for u in [ref] + unseen}
    tr = pd.concat([d5.fenetres_derive(df, u, d5.SCENARIOS[nom_sc])[0] for u in train], ignore_index=True)
    S, W, un, cy = tr[v.S_COLS].values, tr[v.W_COLS].values, tr.unit.values, tr.cycle.values
    ind = r24.indicateurs(S, W, un, cy)
    perf = {}
    for nom, f in methodes(S, W, un, cy).items():
        fen = pd.concat([x.assign(score=f(x[v.S_COLS].values, x[v.W_COLS].values))
                         for x in list(complets.values()) + [tr]], ignore_index=True)
        res = evaluer(fen, ds=ds, train_units=train, ref_unit=ref, non_vues=unseen)
        perf[nom] = float(res[res.role == "non_vue"].corr_spearman.median())
    return [dict(jeu="N-CMAPSS", cas=f"{ds} {nom_sc}", decoupage=f"ref u{ref}", **ind, **perf)]


def tache_cmapss(args):
    fd, cas, retire, tirage = args
    d = b19.a6.charger(fd)
    rng = np.random.default_rng(tirage)
    u = d.unit.unique()
    tr = d[d.unit.isin(set(rng.choice(u, size=len(u) // 2, replace=False)))]
    if retire is not None:
        tr = tr[tr.regime != retire]
    X = [c for c in b19.a6.COLS[5:] if tr[c].std() > 1e-8]
    Wk = [c for c in b19.W_C if tr[c].std() > 1e-8]
    ind = r24.indicateurs(tr[X].values.astype(float), tr[Wk].values.astype(float), tr.unit.values,
                          tr.cycle.values)
    return [dict(jeu="C-MAPSS", cas=f"{fd} {cas}", decoupage=f"tirage {tirage}", tirage=tirage, **ind)]


def perf_cmapss() -> pd.DataFrame:
    r19 = pd.read_csv(v.OUT / "vB19_cmapss.csv")
    r21 = pd.read_csv(v.OUT / "vB21_d_cmapss.csv")
    p = r19.groupby(["FD", "cas", "tirage", "methode"]).sp_cible.median().unstack("methode")
    p = p[["sfa_brut", "sfa_resid_quad"]].rename(columns={"sfa_brut": "lenteur", "sfa_resid_quad": "modele"})
    sw = r21[r21.methode == "span_W v2"].groupby(["FD", "cas", "tirage"]).sp_cible.median().rename("span_W")
    p = p.join(sw).reset_index()
    p["cas"] = p.FD + " " + p.cas
    return p.drop(columns="FD")


def tache_batteries(groupe_tr):
    df = b22.charger()
    grp = {b_: g for g, bs in b22.GROUPES.items() for b_ in bs}
    df["groupe"] = df.bat.map(grp)
    tr = df[df.groupe == groupe_tr]
    S, W, un, cy = tr[b22.S_B].values, tr[b22.W_B].values, tr.bat.values, tr.cycle.values
    ind = r24.indicateurs(S, W, un, cy)
    fs = methodes(S, W, un, cy)
    te = df[df.groupe != groupe_tr]
    sp = {}
    for nom, f in fs.items():
        t = te.assign(s=f(te[b22.S_B].values, te[b22.W_B].values))
        fv = t.groupby(["groupe", "bat", "cycle"]).agg(s=("s", "mean"), cap=("capacite", "first")).reset_index()
        for (g, b_), x in fv.groupby(["groupe", "bat"]):
            sp.setdefault(g, {}).setdefault(nom, []).append(spearmanr(x.s, -x.cap).statistic)
    return [dict(jeu="batteries", cas=g, decoupage=f"entr. {groupe_tr}", **ind,
                 **{nom: float(np.median(x)) for nom, x in d.items()}) for g, d in sp.items()]


# --------------------------------------------------------------------------------------------------
def main():
    tn = []
    for ds in d5.DS_:
        fc = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv", usecols=["unit", "Fc"]).groupby("unit").Fc.first()
        tn += [(ds, sc, ref) for sc in ["aucune", "montante_h0.30"] for ref in sorted(fc[fc == 1].index)]
    tc = [(fd, cas, retire, t) for fd in ["FD002", "FD004"] for cas, retire, _ in b19.a6.CAS
          for t in range(b19.N_TIRAGES)]
    tb = list(b22.GROUPES)
    with Pool(10) as pool:
        rn = pool.map_async(tache_ncmapss, tn, chunksize=1)
        rc = pool.map_async(tache_cmapss, tc, chunksize=1)
        rb = pool.map_async(tache_batteries, tb, chunksize=1)
        rows = [r for rr in rn.get() + rc.get() + rb.get() for r in rr]
    t = pd.DataFrame(rows)
    pc = perf_cmapss()
    c = t[t.jeu == "C-MAPSS"].drop(columns=M, errors="ignore").merge(pc, on=["cas", "tirage"], how="left")
    t = pd.concat([t[t.jeu != "C-MAPSS"], c], ignore_index=True)
    t["meilleure"] = t[M].max(axis=1)
    t["choix"] = [r24.regle(p, D) for p, D in zip(t.p_hors, t.D)]
    t["regret_regle"] = t.meilleure - t.apply(lambda r: r[r.choix], axis=1)
    for m in M:
        t[f"regret_{m}"] = t.meilleure - t[m]
    ref24 = pd.read_csv(v.OUT / "vB24_regle_choix.csv").set_index("cas").choix
    t["choix_vB24"] = [("modele" if j == "batteries" else ref24.get(c_)) for j, c_ in zip(t.jeu, t.cas)]
    t["stable"] = t.choix == t.choix_vB24
    t.to_csv(v.OUT / "vB27_decoupages.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["jeu", "cas", "decoupage", "p_hors", "D", "lenteur", "modele", "span_W", "choix", "regret_regle"]
    print(t[cols].round(2).to_string(index=False))
    print("\n=== par jeu ===")
    g = t.groupby("jeu").agg(decoupages=("cas", "size"), stabilite=("stable", "mean"),
                             p_hors_min=("p_hors", "min"), p_hors_max=("p_hors", "max"),
                             sous_0_3=("p_hors", lambda x: int((x < r24.P_SEUIL).sum())),
                             regret_med=("regret_regle", "median"), regret_max=("regret_regle", "max"))
    print(g.round(3).to_string())
    print("\n=== regret sur tous les découpages : médiane / max / part > 0,2 ===")
    for c_ in ["regret_regle"] + [f"regret_{m}" for m in M]:
        print(f"{c_:14s} médiane {t[c_].median():.3f}   max {t[c_].max():.3f}   > 0,2 : {(t[c_] > 0.2).mean():.1%}")
    print(f"\nstabilité globale du choix : {t.stable.mean():.1%} ({int(t.stable.sum())}/{len(t)})")
    print("\ndécoupages à regret > 0,2 :")
    x = t[t.regret_regle > 0.2][cols]
    print(x.round(2).to_string(index=False) if len(x) else "aucun")


if __name__ == "__main__":
    main()
