"""Itération 10 — brancher NOTRE métrique covariate sur l'indicateur H-CDE reproduit.

LE TEST QUI RESTAIT
-------------------
H-CDE reproduit suit la vraie dégradation (alignment R²=0,915 : la moitié *semantic* est réussie,
avec exogènes). La question ouverte depuis le début : le SOTA fait-il AUSSI la moitié *covariate* ?
C.-à-d. son indicateur de santé reste-t-il plat sur le SAIN quand la condition de vol s'écarte, ou
monte-t-il faussement (fausse alarme covariate, comme un détecteur aveugle à W) ?

CE QU'ON FAIT
-------------
On recharge le modèle entraîné (epoch 78, `model.pt`) et on ré-infère sur TOUTES les fenêtres de
l'unité 9 (non vue) — y compris les SAINES (hs==1), que l'éval du dépôt jette (`df_test[hs==0]`).
Pour chaque fenêtre lente on récupère : l'indicateur H-CDE (PC1 du latent lent), la vraie
dégradation, hs, la condition de vol moyenne W, et les capteurs bruts moyens.

On applique alors la métrique GELÉE d'it2b, à l'identique :
  - seuil = 95e pct des scores du TRAIN-SAIN (fenêtres saines de [1,4,7]) → cible 5 % d'alarme ;
  - sur le TEST-SAIN (fenêtres saines de l'unité 9) : alarme_proche/loin (loin = wd ≥ 90e pct),
    et fuite_covariate = spearman(score, distance-W).
Sur LES MÊMES fenêtres DS01, on compare l'indicateur H-CDE à un détecteur naïf `sans_exo`
(Mahalanobis sur capteurs standardisés) — comparaison apples-to-apples.

Sanity : on recalcule l'alignment_score (leur métrique, [1,4,7]→9 sur hs==0) ; s'il retombe ~0,9,
les poids et le fenêtrage sont bons (le réel tranche).

Usage, depuis Probleme_Shift/code :  <notebook_env python> it10_hcde_covariate.py
Sortie : Probleme_Shift/results/it10_hcde_covariate.{csv,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

HCDE_ROOT = Path(__file__).resolve().parents[1] / "hcde"
sys.path.insert(0, str(HCDE_ROOT))
from src.model.hcde import HCDE  # noqa: E402  (import du modèle seul, pas du cfg)

RUN = HCDE_ROOT / "run/results/ncmapss_lr/hcde_lr/0"
CSV = HCDE_ROOT / "run/datasets/ncmapss/N-CMAPSS_DS01.csv"
OUT = Path(__file__).resolve().parents[1] / "results"

W_VARS = ["alt", "Mach", "TRA", "T2"]
X_VARS = ["T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24",
          "Ps30", "P40", "P50", "Nf", "Nc", "Wf"]
FEATS = W_VARS + X_VARS
TRAIN_UNITS = [1, 4, 7]
TEST_UNIT = 9

# fenêtrage = config du run reproduit (hcde_lr.yaml)
N_WINDOWS, DILATION, STRIDE, HORIZON = 100, 20, 5, 2
FAST_WS, FAST_DIL = 5, 2
FAR_Q = 0.90


def build_model() -> HCDE:
    """HCDE avec exactement les hyperparamètres du run reproduit, poids de l'epoch 78."""
    model = HCDE(
        in_dim=len(FEATS), n_context_vars=len(W_VARS),
        latent_dim_1=5, latent_dim_2=10, path_dim=10, hidden_dim=64, num_layers=2,
        out_dim=len(X_VARS), interpolation="cubic",
        transform_path=True, prodp_1=True, prodp_2=False, gamma=10.0, z_aligment=False,
        regression=True, ing_method="dopri5", adjoint=False,
        variable_length=False, rollout=True,
        fast_time_steps=FAST_WS, epsilon=FAST_DIL / DILATION,
        activation="silu", rtol=1e-3, atol=1e-5,
    )
    ckpt = torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"modèle chargé (best_epoch={ckpt['best_epoch']})")
    return model


def unit_windows(Xu: np.ndarray):
    """Bornes (start, end) des fenêtres lentes d'une unité, mêmes règles que le dépôt."""
    L = len(Xu)
    span = N_WINDOWS * DILATION
    return [(i, i + span) for i in range(0, L, STRIDE) if i + span + HORIZON < L]


@torch.no_grad()
def infer(model: HCDE, scaler: StandardScaler, df: pd.DataFrame, unit: int, batch=256):
    """Ré-infère toutes les fenêtres d'une unité (par lots) ; renvoie un DataFrame par fenêtre."""
    d = df[df.unit == unit].reset_index(drop=True)
    Xraw = d[FEATS].values.astype(np.float32)          # brut (pour W et capteurs)
    Xstd = scaler.transform(Xraw).astype(np.float32)   # standardisé (entrée modèle)
    # `degradation` = norme des paramètres de santé ACTIFS du DS (cf. nc_h5_to_csv.py) ; sur DS01
    # un seul est actif → se réduit à |HPT_eff_mod| (protocole du papier). Repli si CSV ancien.
    col = "degradation" if "degradation" in d.columns else "HPT_eff_mod"
    Draw = np.abs(d[col].values.astype(np.float32))
    hs = d["hs"].values.astype(int)
    cyc = d["cycle"].values.astype(int)

    wins = unit_windows(Xstd)
    xs, xfs, rows = [], [], []
    for (s, e) in wins:
        xs.append(Xstd[s:e:DILATION].T)                                   # (18,100)
        sf, ef = e - FAST_WS * FAST_DIL + FAST_DIL, e + FAST_DIL
        xfs.append(Xstd[sf:ef:FAST_DIL].T)                                # (18,5)
        li = e                                          # last_index (cf. _get_multiscale_item)
        rows.append(dict(
            unit=unit, cycle=int(cyc[li]), hs=int(hs[li]), degradation=float(Draw[li]),
            **{f"W_{k}": float(Xraw[s:e, j].mean()) for j, k in enumerate(W_VARS)},
            **{f"S_{k}": float(Xraw[s:e, len(W_VARS) + j].mean()) for j, k in enumerate(X_VARS)},
        ))
    X = torch.tensor(np.stack(xs), dtype=torch.float32)                   # (N,18,100)
    XF = torch.tensor(np.stack(xfs), dtype=torch.float32)                 # (N,18,5)
    lat = []
    for i in range(0, len(X), batch):
        _, _, d_all, _ = model({"x": X[i:i + batch], "xf": XF[i:i + batch]}, return_latent=True)
        lat.append(d_all[:, -1, :].numpy())            # dernier pas lent, 5-dim
        print(f"  unité {unit}: {min(i + batch, len(X))}/{len(X)} fenêtres", flush=True)
    return pd.DataFrame(rows), np.vstack(lat)


def w_distance(W: np.ndarray, W_ref: np.ndarray) -> np.ndarray:
    mu, sd = W_ref.mean(0), W_ref.std(0) + 1e-9
    return np.linalg.norm((W - mu) / sd, axis=1)


def mahalanobis_fit(Z: np.ndarray):
    mu = Z.mean(0)
    cov = np.cov(Z, rowvar=False) + 1e-6 * np.eye(Z.shape[1])
    P = np.linalg.inv(cov)
    return lambda A: np.sqrt(np.einsum("ij,jk,ik->i", A - mu, P, A - mu))


def auroc(score_pos: np.ndarray, score_neg: np.ndarray) -> float:
    """P(score dégradé > score sain), sans seuil (identique à it1)."""
    s = np.concatenate([score_pos, score_neg])
    r = s.argsort().argsort()
    r_pos = r[: len(score_pos)].sum()
    n1, n2 = len(score_pos), len(score_neg)
    return float((r_pos - n1 * (n1 - 1) / 2) / (n1 * n2))


def fpr_at_tpr(score_pos: np.ndarray, score_neg: np.ndarray, tpr: float = 0.95) -> float:
    """À `tpr` de dégradé (pos) détecté, fraction de sain (neg) au-dessus du seuil (identique à it1)."""
    thr = np.quantile(score_pos, 1 - tpr)
    return float((score_neg >= thr).mean())


def _resid(y, x):
    """Résidu de y après régression linéaire sur x (retrait de l'effet de x)."""
    X1 = np.c_[np.ones_like(x), x]
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    return y - X1 @ b


def covariate_row(name, score_test_heal, wd_test_heal, near, far):
    """Référence = conditions HABITUELLES (proche). Seuil calibré sur le sain-proche du MÊME jeu
    (test) à 5 % ; on lit ensuite la fausse alarme sur le sain-loin (inhabituel). Robuste au
    covariate ⇒ alarme_loin ≈ 5 %. Confond régime/dégradation ⇒ alarme_loin ≫ 5 %."""
    seuil = np.quantile(score_test_heal[near], 0.95)      # 5 % de fausse alarme en conditions habituelles
    al = float((score_test_heal[far] >= seuil).mean())
    return dict(
        detecteur=name,
        alarme_habituel=0.05,                              # par construction (référence)
        alarme_inhabituel=al,
        inflation=al / 0.05,
        fuite_rho=float(spearmanr(score_test_heal, wd_test_heal).statistic),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)

    # scaler standard, ajusté sur w+x de [1,4,7] toutes lignes (= train+val du run)
    scaler = StandardScaler().fit(df[df.unit.isin(TRAIN_UNITS)][FEATS].values)

    cache_m, cache_l = OUT / "it10_infer_meta.csv", OUT / "it10_infer_latent.npy"
    if cache_m.exists() and cache_l.exists():
        meta = pd.read_csv(cache_m); latent = np.load(cache_l)
        print(f"inférence rechargée du cache ({len(meta)} fenêtres) — "
              f"supprimer {cache_l.name} pour relancer")
    else:
        model = build_model()
        metas, lats = [], []
        for u in TRAIN_UNITS + [TEST_UNIT]:
            m, l = infer(model, scaler, df, u)
            m["split"] = "train" if u in TRAIN_UNITS else "test"
            metas.append(m); lats.append(l)
            print(f"unité {u}: {len(m)} fenêtres, saines(hs==1) {int((m.hs == 1).sum())}, "
                  f"dégradées {int((m.hs == 0).sum())}")
        meta = pd.concat(metas, ignore_index=True)
        latent = np.vstack(lats)
        meta.to_csv(cache_m, index=False); np.save(cache_l, latent)

    # indicateur H-CDE = PC1 du latent lent (PCA ajustée sur le train, comme le dépôt),
    # orienté pour corréler positivement avec la vraie dégradation.
    tr = (meta.split == "train").values
    pca = PCA(n_components=3).fit(latent[tr])
    pc = pca.transform(latent)[:, 0]
    if np.corrcoef(pc[tr], meta.degradation.values[tr])[0, 1] < 0:
        pc = -pc
    meta["hcde_index"] = pc

    # --- sanity : alignment_score du dépôt, [1,4,7]->9 sur hs==0 ---
    te_deg = (meta.split == "test").values & (meta.hs == 0).values
    reg = LinearRegression().fit(latent[tr], meta.degradation.values[tr])
    align = r2_score(meta.degradation.values[te_deg], reg.predict(latent[te_deg]))
    pc1_corr_u9 = np.corrcoef(pc[te_deg], meta.degradation.values[te_deg])[0, 1]
    print(f"\n[sanity] alignment_score [1,4,7]->9 (hs==0) = {align:.3f} "
          f"(publié 0,964 ; reproduit ~0,915) | PC1 corr u9(hs0) = {pc1_corr_u9:+.3f}")

    meta.to_csv(OUT / "it10_hcde_covariate_windows.csv", index=False)

    # --- masques : unité 9 (non vue), sain proche / sain loin / dégradé ---
    trh = tr & (meta.hs == 1).values                              # train-sain (réf. distance-W)
    teh = (meta.split == "test").values & (meta.hs == 1).values   # test(u9)-sain
    ted = (meta.split == "test").values & (meta.hs == 0).values   # test(u9)-dégradé (semantic)
    print(f"\nunité 9 : sain {int(teh.sum())}, dégradé {int(ted.sum())}")

    Wtr = meta[[f"W_{k}" for k in W_VARS]].values
    wd_te = w_distance(Wtr[teh], Wtr[trh])       # distance-W du sain u9 à la condition d'entraînement
    far = wd_te >= np.quantile(wd_te, FAR_Q)     # sain-LOIN = conditions inhabituelles (top 10 %)
    near = ~far                                  # sain-PROCHE = conditions habituelles (référence)

    # scores des trois détecteurs sur TOUTES les fenêtres u9 (sain+dégradé)
    Sall = meta[[f"S_{k}" for k in X_VARS]].values
    s_scaler = StandardScaler().fit(Sall[trh])
    maha = mahalanobis_fit(s_scaler.transform(Sall[trh]))
    scores = {
        "sans_exo (naïf window-level)": maha(s_scaler.transform(Sall)),
        "H-CDE (indicateur reproduit)": pc,
    }
    # NOTE — l'horloge (cycle) est ABANDONNÉE comme baseline (décision Martin, 15 sept.) : les
    # trajectoires N-CMAPSS sont run-to-failure, donc le temps « détecte » trivialement, MAIS aucun
    # de nos modèles n'a le temps en entrée ni le moyen de savoir où il en est dans la trajectoire.
    # L'horloge ne peut donc pas être jouée par eux : l'afficher n'informe sur rien et brouille la
    # lecture. Les métriques full-spectrum restent valides pour ces modèles.

    # ========================================================================================
    # (A) MÉTRIQUE DÉCLARÉE — full-spectrum (it1), sur l'unité 9. Référence des fausses alarmes =
    #     conditions HABITUELLES (sain-proche).
    # ========================================================================================
    heal_near = np.where(teh)[0][near]           # indices absolus
    heal_far = np.where(teh)[0][far]
    fs_rows = []
    for name, sc in scores.items():
        deg, hn, hf = sc[ted], sc[heal_near], sc[heal_far]
        fs_rows.append(dict(
            detecteur=name,
            AUROC=auroc(deg, sc[teh]),
            FPR95_habituel=fpr_at_tpr(deg, hn),          # référence
            FPR95_inhabituel=fpr_at_tpr(deg, hf),        # covariate shift « dur »
            surcout_covariate=fpr_at_tpr(deg, hf) / max(fpr_at_tpr(deg, hn), 1e-6),
        ))
    fs = pd.DataFrame(fs_rows)
    fs.to_csv(OUT / "it10_fullspectrum.csv", index=False)
    print("\n(A) Full-spectrum sur l'unité 9 — FPR@95 : réf = conditions HABITUELLES vs inhabituelles")
    print(fs.round(3).to_string(index=False))
    print("    Robuste au covariate ⇒ FPR95_inhabituel ≈ FPR95_habituel. Confond ⇒ inhabituel ≫ habituel.")

    # ========================================================================================
    # (B) Fausses alarmes sur le SAIN seul (it2b), référence = habituel : seuil calibré à 5 % sur
    #     le sain-proche, lecture sur le sain-loin. Zéro biais run-to-failure.
    # ========================================================================================
    res = pd.DataFrame([
        covariate_row(n, scores[n][teh], wd_te, near, far)
        for n in ["sans_exo (naïf window-level)", "H-CDE (indicateur reproduit)"]
    ])
    res.to_csv(OUT / "it10_hcde_covariate.csv", index=False)
    print("\n(B) Sain seul — seuil calibré à 5 % sur conditions HABITUELLES ; lecture sur inhabituelles")
    print(res.round(3).to_string(index=False))

    # --- figure : (gauche) distributions full-spectrum H-CDE ; (droite) score sain vs distance-W ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    scH = scores["H-CDE (indicateur reproduit)"]
    lo, hi = np.quantile(scH[teh | ted], [0.01, 0.99])
    b = np.linspace(lo, hi, 50)
    ax[0].hist(scH[heal_near], b, density=True, histtype="step", lw=1.8, label="sain proche (habituel)")
    ax[0].hist(scH[heal_far], b, density=True, histtype="step", lw=1.8, label="sain LOIN (inhabituel)")
    ax[0].hist(scH[ted], b, density=True, histtype="step", lw=1.8, label="dégradé (semantic)")
    rH = fs[fs.detecteur.str.startswith("H-CDE")].iloc[0]
    ax[0].set_title(f"H-CDE — full-spectrum (unité 9)\nAUROC {rH.AUROC:.2f} | "
                    f"FPR95 habituel {rH.FPR95_habituel:.2f} → inhabituel {rH.FPR95_inhabituel:.2f}",
                    fontsize=10)
    ax[0].set_xlabel("indicateur de santé H-CDE"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

    scv = scH[teh]
    q = np.linspace(0, 1, 16); edges = np.quantile(wd_te, q); mid = 0.5 * (edges[:-1] + edges[1:])
    med = [np.median(scv[(wd_te >= edges[i]) & (wd_te < edges[i + 1])]) for i in range(len(edges) - 1)]
    ax[1].scatter(wd_te, scv, s=6, alpha=.25, color="0.6")
    ax[1].plot(mid, med, "o-", color="C3", lw=1.8, label="médiane par tranche")
    ax[1].axvline(np.quantile(wd_te, FAR_Q), color="0.4", ls="--", lw=1, label="seuil proche/loin")
    ax[1].set_title(f"H-CDE — indice sur le SAIN vs condition\nrho(indice, distance-W) = "
                    f"{spearmanr(scv, wd_te).statistic:+.2f}", fontsize=10)
    ax[1].set_xlabel("distance-W  (condition de vol inhabituelle →)")
    ax[1].set_ylabel("indice de santé (sain)"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.suptitle("It10 — le SOTA fait-il la moitié covariate ? (unité 9 non vue) — réf = conditions "
                 "habituelles", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "it10_hcde_covariate.png", dpi=130)
    print(f"\nfigure : {OUT / 'it10_hcde_covariate.png'}")


if __name__ == "__main__":
    main()
