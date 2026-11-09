"""Itération 12 — évaluation de la campagne multi-DS / multi-graines. PRODUIT LA MATRICE, NE CONCLUT PAS.

Pour chaque (DS, graine), on recharge le modèle entraîné par `campagne_ds.py` et on mesure, sur
chaque unité d'évaluation, LES DEUX MOITIÉS — protocole identique à it11 :

  covariate : fausse alarme sur le SAIN, seuil calibré à 5 % sur le sain de l'unité de RÉFÉRENCE
              (même classe de vol que l'entraînement = conditions HABITUELLES, cf. Martin).
              Robuste ⇒ ~5 % partout, y compris sur une classe de vol jamais vue.
  semantic  : amplitude = indice(dégradé) − indice(sain) — la RÉPONSE à une vraie dégradation ;
              + corr(indice, vraie dégradation), alignment R² (transfert depuis le train), AUROC, FPR@95.
              On rapporte aussi l'ampleur VRAIE de dégradation, pour vérifier que la comparaison
              d'amplitude entre unités est équitable (dégradations comparables).

Aucune horloge (décision Martin : les modèles n'ont pas le temps en entrée).
Aucune interprétation ici : on écrit `results/it12_campagne.csv`.

Usage : <torch python> it12_campagne_eval.py [--ds DS01 DS02 ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import it10_hcde_covariate as it10  # noqa: E402  (DILATION lu à l'appel : on le fixe par run)
from campagne_ds import HCDE, KEYS, PLAN, parse_key  # noqa: E402
from it10_hcde_covariate import (  # noqa: E402
    FEATS, N_WINDOWS, OUT, W_VARS, X_VARS, auroc, fpr_at_tpr, infer, mahalanobis_fit, unit_windows,
)


def window_meta(df: pd.DataFrame, unit: int) -> pd.DataFrame:
    """Métadonnées par fenêtre, SANS modèle (mêmes bornes de fenêtre qu'`infer`).

    Le latent est coûteux et ne dépend QUE des capteurs+W (inchangés) ; les métadonnées sont
    dérivées et bon marché. On ne met donc en cache que le latent, et on recalcule ceci à chaque
    fois — c'est ce qui permet de corriger la cible `degradation` sans réinférer.
    """
    d = df[df.unit == unit].reset_index(drop=True)
    Xraw = d[FEATS].values.astype(np.float32)
    col = "degradation" if "degradation" in d.columns else "HPT_eff_mod"
    Draw = np.abs(d[col].values.astype(np.float32))
    hs, cyc = d["hs"].values.astype(int), d["cycle"].values.astype(int)
    rows = []
    for (s, e) in unit_windows(Xraw):
        rows.append(dict(
            unit=unit, cycle=int(cyc[e]), hs=int(hs[e]), degradation=float(Draw[e]),
            **{f"W_{k}": float(Xraw[s:e, j].mean()) for j, k in enumerate(W_VARS)},
            **{f"S_{k}": float(Xraw[s:e, len(W_VARS) + j].mean()) for j, k in enumerate(X_VARS)},
        ))
    return pd.DataFrame(rows)


DATASETS = HCDE / "run/datasets/ncmapss"


def load_model(ckpt: Path):
    """Même architecture que la config reproduite (identique pour tous les DS)."""
    from it10_hcde_covariate import build_model as _bm
    import it10_hcde_covariate as m
    old = m.RUN
    m.RUN = ckpt.parent            # build_model lit RUN/'model.pt'
    try:
        return _bm()
    finally:
        m.RUN = old


def eval_run(key: str, seed: int, prefix: str = "camp") -> list[dict]:
    """`key` = DS (dilation du papier) ou variante `DSxx_d<dilation>` (fenêtre adaptée aux vols longs)."""
    ds, dil = parse_key(key)
    it10.DILATION = dil      # fenêtrage + epsilon du modèle, identiques à l'entraînement de ce run
    fc_seen, train_units, ref_unit, unseen = PLAN[ds]
    # v1 (`camp`) : un dossier par DS, graines en sous-dossiers ; v3 (`c3`…) : un dossier par (DS, graine)
    n = f"camp_{key}" if prefix == "camp" else f"{prefix}_{key}_s{seed}"
    run_dir = HCDE / f"run/results/{n}/{n}/{seed}"
    ckpt = run_dir / "model.pt"
    if not ckpt.exists():
        print(f"  {key} graine {seed} : pas de model.pt — ignoré")
        return []
    # GARDE-FOU : ne JAMAIS évaluer un entraînement en cours. `model.pt` existe dès la 1re epoch
    # (meilleur modèle courant) ; l'évaluer donnerait des chiffres de modèle non entraîné, qui
    # ressemblent à un résultat. On exige le checkpoint de la dernière epoch.
    import yaml
    cfg = yaml.safe_load((HCDE / f"run/configs/ncmapss/{n}.yaml").read_text())
    max_ep = cfg["optim"]["max_epochs"]
    assert cfg["dataset"]["dilation"] == dil, f"{key}: dilation config {cfg['dataset']['dilation']} != {dil}"
    if not (run_dir / f"ckpt/{max_ep - 1}.ckpt").exists():
        done_ep = torch.load(ckpt, map_location="cpu", weights_only=False).get("best_epoch", "?")
        print(f"  {key} graine {seed} : entraînement NON TERMINÉ "
              f"(meilleure epoch {done_ep}/{max_ep - 1}) — ignoré")
        return []

    df = pd.read_csv(DATASETS / f"N-CMAPSS_{ds}.csv")
    fc_of = df.groupby("unit")["Fc"].first().to_dict()
    scaler = StandardScaler().fit(df[df.unit.isin(train_units)][FEATS].values)
    model = load_model(ckpt)

    units = list(train_units) + [ref_unit] + list(unseen)
    # métadonnées : TOUJOURS recalculées depuis le CSV courant (bon marché, et c'est ce qui porte
    # la cible `degradation`)
    metas = []
    for u in units:
        m_u = window_meta(df, u)
        m_u["role"] = "train" if u in train_units else ("ref" if u == ref_unit else "non_vu")
        metas.append(m_u)
    meta = pd.concat(metas, ignore_index=True)

    # latent : seul élément coûteux, mis en cache (ne dépend que des capteurs+W, inchangés)
    tag = key if prefix == "camp" else f"{prefix}_{key}"
    cache = OUT / f"it12_latent_{tag}_s{seed}.npy"
    if cache.exists():
        latent = np.load(cache)
        if len(latent) != len(meta):      # fenêtrage incohérent → on refait
            print(f"  {ds} : cache latent désaligné ({len(latent)} vs {len(meta)}) — réinférence")
            cache.unlink()
    if not cache.exists():
        lats = []
        for u in units:
            _, l = infer(model, scaler, df, u)
            lats.append(l)
        latent = np.vstack(lats)
        np.save(cache, latent)
    assert len(latent) == len(meta), f"{ds}: latent {len(latent)} != meta {len(meta)}"

    tr = (meta.role.values == "train")
    pca = PCA(n_components=3).fit(latent[tr])
    pc = pca.transform(latent)[:, 0]
    if np.corrcoef(pc[tr], meta.degradation.values[tr])[0, 1] < 0:
        pc = -pc
    reg = LinearRegression().fit(latent[tr], meta.degradation.values[tr])

    S = meta[[f"S_{k}" for k in X_VARS]].values
    trh = tr & (meta.hs == 1).values
    ss = StandardScaler().fit(S[trh])
    naive = mahalanobis_fit(ss.transform(S[trh]))(ss.transform(S))

    ref_h = (meta.unit.values == ref_unit) & (meta.hs.values == 1)
    thr_h, thr_n = np.quantile(pc[ref_h], 0.95), np.quantile(naive[ref_h], 0.95)

    # portée de la fenêtre lente en nombre de VOLS (le confondant corrigé le 16 sept.)
    s_tr = df[df.unit.isin(train_units)]
    vols_fenetre = N_WINDOWS * dil / (len(s_tr) / s_tr.groupby("unit").cycle.nunique().sum())

    rows = []
    for u in [ref_unit] + list(unseen):
        m_u = meta.unit.values == u
        heal, deg = m_u & (meta.hs.values == 1), m_u & (meta.hs.values == 0)
        if heal.sum() < 50 or deg.sum() < 50:
            print(f"  {ds} u{u} : trop peu de fenêtres (sain {heal.sum()}, dég {deg.sum()}) — ignoré")
            continue
        d = meta.degradation.values
        rows.append(dict(
            run=key, DS=ds, dilation=dil, vols_par_fenetre=round(float(vols_fenetre), 2),
            graine=seed, unite=u, Fc=int(fc_of[u]), Fc_vue=fc_seen,
            classe=("REF (classe vue)" if u == ref_unit else "NON VUE"),
            n_sain=int(heal.sum()), n_degrade=int(deg.sum()),
            # équité de la comparaison d'amplitude
            ampleur_vraie=float(np.median(d[deg]) - np.median(d[heal])),
            # semantic
            amplitude_indice=float(np.median(pc[deg]) - np.median(pc[heal])),
            corr_indice_degr=float(np.corrcoef(pc[m_u], d[m_u])[0, 1]),
            alignment_R2=float(r2_score(d[deg], reg.predict(latent[deg]))),
            AUROC=float(auroc(pc[deg], pc[heal])),
            FPR95=float(fpr_at_tpr(pc[deg], pc[heal])),
            # covariate
            fausse_alarme_HCDE=float((pc[heal] >= thr_h).mean()),
            fausse_alarme_naif=float((naive[heal] >= thr_n).mean()),
        ))
    return rows


def main(keys, seeds, prefix="camp"):
    allrows = []
    for key in keys:
        for s in seeds:
            print(f"--- {key} graine {s} ---", flush=True)
            allrows += eval_run(key, s, prefix)
    if not allrows:
        print("aucun run évaluable pour l'instant."); return
    res = pd.DataFrame(allrows)
    out = OUT / ("it12_campagne.csv" if prefix == "camp" else f"{prefix}_eval.csv")
    if out.exists():          # fusionner avec l'existant (campagne incrémentale)
        old = pd.read_csv(out)
        if "run" not in old.columns:          # lignes d'avant les clés de run = dilation du papier
            old = old.assign(run=old.DS, dilation=20)
        res = (pd.concat([old, res])
                 .drop_duplicates(subset=["run", "graine", "unite"], keep="last")
                 .sort_values(["run", "graine", "unite"]))
    res.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print("\n=== MATRICE (brute, sans interprétation) ===")
    print(res.round(3).to_string(index=False))
    print(f"\nécrit : {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=KEYS, help="DS01 … ou variantes DS02_d53 …")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--prefix", default="camp", help="camp = campagne v1 ; c3 = campagne v3")
    a = ap.parse_args()
    main(a.keys, a.seeds, a.prefix)
