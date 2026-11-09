"""vB35 — Table unifiée du papier B (Tab. 3, Fig. 3) : toutes les méthodes sur les mêmes cas.

Consolidation, AUCUNE nouvelle revendication : les méthodes et les cas existent déjà (vB19–vB34) mais sur des cas
partiellement différents. Ici, chaque méthode est évaluée sur exactement les mêmes 124 cas :
  - N-CMAPSS DS01/05/07, sans dérive fabriquée (la dérive est au § 5.4), 4 rotations Fc1 → 12 cas (vérité :
    dégradation, par vol, évaluateur commun) ;
  - C-MAPSS FD002/FD004 × 6 cas de a6 × 5 tirages → 60 (âge, sur le régime cible) ;
  - batteries : 5 groupes d'entraînement × 4 groupes de test → 20 (capacité) ;
  - roulements de Paderborn : 12 (âge) ; capteurs de gaz : 12 (lot) ; PVDAQ inter-sites : 8 (âge ; non informatif).
XJTU-SY n'est pas dans la table (contexte constant par roulement) : dépôt public.
Méthodes, par famille (définitions inchangées) :
  - sans contexte : `lenteur` (SFA), `tendance_moy`, `tendance_reg` (vB32) ;
  - retrait du contexte : `resid_lin` (SFA sur S − modèle linéaire de W), `resid_quad` (vB24 « modele » ; colonnes
    de résidu sans variance retirées, règle de vB19),
    `residu_FNN` (vB26), `norm_regime` (vB33) ;
  - invariance au contexte : `span_W` v2 (DB-08), `invariant_regime` (vB33), `DANN` (vB34, relu dans
    vB34_dann_regime.csv, médiane de 3 graines) ;
  - variable latente avec contexte : `bajarunas_corr`, `bajarunas_mono` (réimplémentation, vB26).
Réseaux de vB26 : réglages de vB26 inchangés (60 epochs, 2 × 64), 3 graines (médiane) ; seule différence déclarée :
entraînement sous-échantillonné uniformément à 50 000 lignes au plus (graine fixe), pour PVDAQ et les roulements.
Contrôle de cohérence (pas un critère) : les méthodes déjà mesurées sur ces cas doivent retrouver leurs valeurs
(lenteur, tendances : vB32 ; norm/invariant : vB33).

Machine : Mac (Martin, 12 oct.). Sortie : voie_B_avec_exo/results/vB35_*.csv, vB35.txt
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
import sfa  # noqa: E402

FAMILLES = {
    "sans contexte": ["lenteur", "tendance_moy", "tendance_reg"],
    "retrait": ["resid_lin", "resid_quad", "residu_FNN", "norm_regime"],
    "invariance": ["span_W", "invariant_regime", "DANN"],
    "latent avec contexte": ["bajarunas_corr", "bajarunas_mono"],
}
GRAINES_NN, MAX_NN = (0, 1, 2), 50_000


def toutes(S, W, unit, cycle, labels=None):
    """Tous les scoreurs (S_, W_) → score, sauf DANN (relu dans vB34)."""
    import vB26_baselines as b26
    import vB27_robustesse_decoupages as r27
    import vB32_tendance_contre_lenteur as t32
    import vB33_labels_regime as m33
    unit, cycle = np.asarray(unit), np.asarray(cycle)
    out = {}
    for nom, f in t32.methodes(S, unit, cycle).items():
        out[nom] = (lambda S_, W_, f=f: f(S_))
    import vB24_regle_choix as r24
    out["span_W"] = r27.v8.span_W(S, W, unit, cycle, reduire=False)[0].score
    out["span_W"] = (lambda S_, W_, f=out["span_W"]: f(S_))
    muW, sdW = W.mean(0), W.std(0) + 1e-12
    # résidus : colonnes sans variance à l'entraînement retirées (capteurs entièrement expliqués par W, ex. 6
    # capteurs de C-MAPSS) — même règle que vB19 : écart-type > 1e-6 × le plus grand
    for nom, base in [("resid_lin", lambda W_: (W_ - muW) / sdW), ("resid_quad", lambda W_: r24.phi((W_ - muW) / sdW))]:
        reg = LinearRegression().fit(base(W), S)
        E = S - reg.predict(base(W))
        sdE = E.std(0)
        garde = sdE > 1e-6 * sdE.max()
        m = sfa.ajuster(E[:, garde], unit, cycle)
        out[nom] = (lambda S_, W_, m=m, reg=reg, base=base, garde=garde: m.score((S_ - reg.predict(base(W_)))[:, garde]))
    g = m33.methodes(S, W, unit, cycle, labels=labels)
    out["norm_regime"], out["invariant_regime"] = g["norm_regime"], g["invariant_regime"]
    rang = pd.Series(cycle).groupby(unit).rank(pct=True).values
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(S), MAX_NN, replace=False)) if len(S) > MAX_NN else np.arange(len(S))
    nn = {k: [] for k in ["residu_FNN", "bajarunas_corr", "bajarunas_mono"]}
    for gr in GRAINES_NN:
        for k, f in b26.entrainer(S[sel], W[sel], unit[sel], rang[sel], gr).items():
            nn[k].append(f)
    for k, fs in nn.items():
        out.update({f"{k}_g{i}": f for i, f in enumerate(fs)})
    return out


def agreger(row):
    for k in ["residu_FNN", "bajarunas_corr", "bajarunas_mono"]:
        row[k] = float(np.median([row.pop(f"{k}_g{i}") for i in range(len(GRAINES_NN))]))
    return row


# --------------------------------------------------------------------------------------------------
def t_ncmapss(args):
    from evaluateur import evaluer
    ds, ref = args
    v.it10.DILATION = v.parse_key(ds)[1]
    df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
    fc = df.groupby("unit").Fc.first()
    train = [u for u in sorted(fc[fc == 1].index) if u != ref]
    unseen = sorted(fc[fc != 1].index)
    complets = {u: v.fenetres(df, u) for u in [ref] + unseen}
    import vB05_derive_usage as d5
    tr = pd.concat([d5.fenetres_derive(df, u, None)[0] for u in train], ignore_index=True)   # = vB32 « aucune »
    row = dict(jeu="N-CMAPSS", cas=f"{ds} aucune ref u{ref}")
    for nom, f in toutes(tr[v.S_COLS].values, tr[v.W_COLS].values, tr.unit.values, tr.cycle.values).items():
        fen = pd.concat([x.assign(score=f(x[v.S_COLS].values, x[v.W_COLS].values))
                         for x in list(complets.values()) + [tr]], ignore_index=True)
        res = evaluer(fen, ds=ds, train_units=train, ref_unit=ref, non_vues=unseen)
        row[nom] = float(res[res.role == "non_vue"].corr_spearman.median())
    return [agreger(row)]


def t_cmapss(args):
    import vB19_cmapss_avec_w as b19
    fd, cas, retire, cible, tirage = args
    d = b19.a6.charger(fd)
    rng = np.random.default_rng(tirage)
    u = d.unit.unique()
    tr_u = set(rng.choice(u, size=len(u) // 2, replace=False))
    tr = d[d.unit.isin(tr_u)]
    if retire is not None:
        tr = tr[tr.regime != retire]
    te = d[~d.unit.isin(tr_u)]
    X = [c for c in b19.a6.COLS[5:] if tr[c].std() > 1e-8]
    Wk = [c for c in b19.W_C if tr[c].std() > 1e-8]
    row = dict(jeu="C-MAPSS", cas=f"{fd} {cas} tirage {tirage}")
    fs = toutes(tr[X].values.astype(float), tr[Wk].values.astype(float), tr.unit.values, tr.cycle.values,
                labels=tr.regime.values)
    for nom, f in fs.items():
        t = te.assign(s=f(te[X].values.astype(float), te[Wk].values.astype(float)))
        sp = [spearmanr(g[g.regime == cible].cycle, g[g.regime == cible].s).statistic
              for _, g in t.groupby("unit") if (g.regime == cible).sum() >= 10]
        row[nom] = float(np.median(sp))
    return [agreger(row)]


def t_batteries(g_tr):
    import vB22_batteries as b22
    df = b22.charger()
    df["groupe"] = df.bat.map({b: g for g, bs in b22.GROUPES.items() for b in bs})
    tr, te = df[df.groupe == g_tr], df[df.groupe != g_tr]
    fs = toutes(tr[b22.S_B].values, tr[b22.W_B].values, tr.bat.values, tr.cycle.values)
    sp = {}
    for nom, f in fs.items():
        t = te.assign(s=f(te[b22.S_B].values, te[b22.W_B].values))
        fv = t.groupby(["groupe", "bat", "cycle"]).agg(s=("s", "mean"), cap=("capacite", "first")).reset_index()
        for (g, b), x in fv.groupby(["groupe", "bat"]):
            sp.setdefault(g, {}).setdefault(nom, []).append(spearmanr(x.s, -x.cap).statistic)
    return [agreger(dict(jeu="batteries", cas=f"entr. {g_tr} → {g}",
                         **{n: float(np.median(x)) for n, x in d.items()})) for g, d in sp.items()]


def t_roulements(g_tr):
    import vB29_roulements_upb as u29
    df = u29.charger()
    tr = df[df.groupe == g_tr]
    Wk = [c for c in u29.W_U if c == "T_amb" or tr[f"set_{c}"].nunique() > 1]
    fs = toutes(tr[u29.S_U].values, tr[Wk].values, tr.unit.values, tr.cycle.values)
    rows = []
    for g_te in u29.GROUPES:
        if g_te == g_tr:
            continue
        te = df[(df.groupe == g_te) & ~df.unit.isin(u29.ENTR_SEULEMENT)].reset_index(drop=True)
        row = dict(jeu="roulements", cas=f"{g_tr} → {g_te}")
        for nom, f in fs.items():
            t_ = te.assign(s=f(te[u29.S_U].values, te[Wk].values))
            row[nom] = float(np.median([spearmanr(g.cycle, g.s).statistic for _, g in t_.groupby("unit")]))
        rows.append(agreger(row))
    return rows


def t_gaz(args):
    import vB30_capteurs_gaz as g30
    gaz, impairs = args
    df = g30.charger()
    dg = df[df.gaz == gaz]
    med = dg.conc.median()
    u_tr = [u for u in range(1, 17) if (u % 2 == 1) == impairs]
    tr = dg[dg.unit.isin(u_tr) & (dg.conc <= med)]
    te = dg[~dg.unit.isin(u_tr) & (dg.conc > med)].reset_index(drop=True)
    row = dict(jeu="gaz", cas=f"gaz {gaz} {'impairs→pairs' if impairs else 'pairs→impairs'}")
    for nom, f in toutes(tr[g30.S_G].values, tr[["logc"]].values, tr.unit.values, tr.cycle.values).items():
        t_ = te.assign(s=f(te[g30.S_G].values, te[["logc"]].values))
        row[nom] = float(np.median([spearmanr(g.lot, g.s).statistic for _, g in t_.groupby("unit")]))
    return [agreger(row)]


def t_pvdaq(args):
    import vB31_pvdaq as p31
    a, b, out_u = args
    df = p31.ch.charger()
    df = df[df.unit.isin(sum(p31.SITES.values(), []))].copy()
    df["age"] = (df.jour - df.groupby("unit").jour.transform("min")).dt.days
    tr = df[df.unit.isin([u for u in p31.SITES[a] if u != out_u])]
    te = df[df.unit.isin(p31.SITES[b])].reset_index(drop=True)
    row = dict(jeu="PVDAQ", cas=f"{a}→{b} sans {out_u}")
    for nom, f in toutes(tr[p31.S_P].values, tr[p31.W_P].values, tr.unit.values, tr.cycle.values).items():
        j = te.assign(s=f(te[p31.S_P].values, te[p31.W_P].values)).groupby(["unit", "age"]).s.mean().reset_index()
        row[nom] = float(np.median([spearmanr(g.age, g.s).statistic for _, g in j.groupby("unit")]))
    return [agreger(row)]


def lancer(t):
    nom, args = t
    return {"N": t_ncmapss, "C": t_cmapss, "B": t_batteries, "R": t_roulements, "G": t_gaz, "P": t_pvdaq}[nom](args)


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
               for t in range(b19.N_TIRAGES)]
    taches += [("B", g) for g in b22.GROUPES] + [("R", g) for g in u29.GROUPES]
    taches += [("G", (gz, imp)) for gz in range(1, 7) for imp in (True, False)]
    taches += [("P", (a, b, u)) for a, b in [("Las_Vegas", "Golden"), ("Golden", "Las_Vegas")] for u in p31.SITES[a]]
    taches = [t for t in taches if t[0] in ("P", "R", "N")] + [t for t in taches if t[0] not in ("P", "R", "N")]
    with Pool(10) as pool:
        t = pd.DataFrame([r for rr in pool.map(lancer, taches, chunksize=1) for r in rr])
    d34 = pd.read_csv(v.OUT / "vB34_dann_regime.csv")[["jeu", "cas", "dann_lent"]].rename(columns={"dann_lent": "DANN"})
    t = t.merge(d34, on=["jeu", "cas"], how="left")
    for fam, ms in FAMILLES.items():
        t[f"meilleure {fam}"] = t[ms].max(axis=1)
    t["meilleure avec contexte"] = t[sum(list(FAMILLES.values())[1:], [])].max(axis=1)
    t.to_csv(v.OUT / "vB35_table_unifiee.csv", index=False)

    pd.set_option("display.width", 300)
    ordre = ["N-CMAPSS", "C-MAPSS", "batteries", "roulements", "gaz", "PVDAQ"]
    ms = sum(FAMILLES.values(), [])
    print("=== Tab. 3 : médiane par jeu (Spearman contre la vérité du jeu) ===")
    tab = t.groupby("jeu")[ms].median().reindex(ordre).T
    tab["cas"] = ""
    print(t.groupby("jeu").size().reindex(ordre).to_string())
    print(tab.drop(columns="cas").round(2).to_string())
    print("\n=== Fig. 3 : meilleure sans contexte contre meilleure avec contexte (médiane par jeu ; cas où sans ≥ avec − 0,05) ===")
    s, a = t["meilleure sans contexte"], t["meilleure avec contexte"]
    f3 = pd.DataFrame({"sans": s, "avec": a, "sans_ok": s >= a - 0.05, "jeu": t.jeu}).groupby("jeu").agg(
        sans=("sans", "median"), avec=("avec", "median"), part_sans_ok=("sans_ok", "mean")).reindex(ordre)
    print(f3.round(2).to_string())
    print("\n=== contrôle de cohérence (médianes) : vB32 / vB33 sur les mêmes cas ===")
    t32 = pd.read_csv(v.OUT / "vB32_tendance_contre_lenteur.csv")
    t33 = pd.read_csv(v.OUT / "vB33_labels_regime.csv")
    c = t.merge(t32[["jeu", "cas", "lenteur", "tendance_reg"]], on=["jeu", "cas"], suffixes=("", "_32"))
    c = c.merge(t33[["jeu", "cas", "norm_regime"]], on=["jeu", "cas"], suffixes=("", "_33"))
    print("écart max |lenteur − vB32| :", round(float((c.lenteur - c.lenteur_32).abs().max()), 4),
          " |tendance_reg − vB32| :", round(float((c.tendance_reg - c.tendance_reg_32).abs().max()), 4),
          " |norm_regime − vB33| :", round(float((c.norm_regime - c.norm_regime_33).abs().max()), 4))


if __name__ == "__main__":
    main()
