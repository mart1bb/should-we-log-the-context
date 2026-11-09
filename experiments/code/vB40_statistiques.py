"""vB40 — Statistiques du papier (R9 de PAPIER_ICDE § 8) : tests appariés, bootstrap hiérarchique, différences critiques.

Aucune nouvelle mesure : relit results/vB39_par_unite.csv (Spearman par unité de test ; réseaux : par graine).
Jeux : N-CMAPSS, C-MAPSS retenu (30 cas), C-MAPSS contrôle (30 cas, rapporté à part), batteries, roulements, gaz,
PVDAQ (non informatif, rapporté sans test). Méthodes : les 12 du tableau III (les méthodes de R10/R12 sont testées
de la même façon, colonnes en plus ; H-CDE de vB44 ajouté sur N-CMAPSS quand il existe, ce qui refait la correction de
Holm de ce jeu ; idem pour V-REx β = 10 de vB45, jeux informatifs ; le diagramme de différences critiques garde les 12 méthodes du banc initial).

Déclaré AVANT de regarder les sorties :
  1. Gras du tableau III : par jeu, méthode de meilleure médiane = référence ; chaque autre méthode est comparée à
     la référence par un test de Wilcoxon signé apparié par cas (bilatéral), p corrigées par Holm sur les méthodes
     du jeu ; gras = non significativement différente de la référence (p corrigée ≥ 0,05) ou référence elle-même.
  2. Contrastes du texte, entre représentants fixés d'avance (ceux de la Fig. 3 : lenteur ; résidu quadratique ;
     span_W ; Bajarunas corr) : différence appariée par unité (réseaux : médiane des graines), médiane par cas, puis
     médiane par jeu ; IC à 95 % par bootstrap hiérarchique (2 000 tirages ; cas tirés avec remise dans le jeu, puis
     unités tirées avec remise dans chaque cas ; graine 0) ; percentiles 2,5 et 97,5.
  3. Diagramme de différences critiques (Demšar 2006) : rangs moyens des 12 méthodes sur les cas hors support des
     jeux informatifs (N-CMAPSS, C-MAPSS retenu, batteries, roulements, gaz : 86 cas), test de Friedman puis
     distance critique de Nemenyi (α = 0,05). Comme C-MAPSS pèse 30 cas sur 86, on rapporte aussi les rangs moyens
     avec un poids égal par jeu (moyenne des rangs moyens par jeu), sans test.
Réserve déclarée : les cas d'un même jeu partagent des unités d'entraînement ou de test ; ni le Wilcoxon par cas ni le
bootstrap ne modélisent cette dépendance (le bootstrap par unités la prend en partie en compte). Les p sont à lire
comme des indications.

Sortie : results/vB40.txt, vB40_tests.csv, vB40_contrastes.csv ; papier/tab_main.tex (via vB38) ; figures/fig5_cd.pdf/png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon

ICI = Path(__file__).resolve().parent.parent
OUT = ICI / "results"
METH12 = ["lenteur", "tendance_moy", "tendance_reg", "resid_lin", "resid_quad", "residu_FNN", "norm_regime",
          "span_W", "invariant_regime", "DANN", "bajarunas_corr", "bajarunas_mono"]
REPR = ["lenteur", "resid_quad", "span_W", "bajarunas_corr"]
JEUX = ["N-CMAPSS", "C-MAPSS retenu", "C-MAPSS contrôle", "batteries", "roulements", "gaz", "PVDAQ"]
INFORMATIFS = ["N-CMAPSS", "C-MAPSS retenu", "batteries", "roulements", "gaz"]
B, ALPHA = 2000, 0.05


def charger():
    L = pd.read_csv(OUT / "vB39_par_unite.csv")
    cm = L.jeu == "C-MAPSS"
    L.loc[cm & L.cas.str.contains(r" R\d"), "jeu"] = "C-MAPSS retenu"
    L.loc[cm & L.cas.str.contains("C0_"), "jeu"] = "C-MAPSS contrôle"
    # réseaux : médiane des graines par unité (pour les contrastes) ; par cas : comme vB35 (médiane des graines des
    # médianes par cas)
    U = L.groupby(["jeu", "cas", "methode", "unite"]).rho.median().unstack("methode").reset_index()
    c = L.groupby(["jeu", "cas", "methode", "graine"]).rho.median().reset_index()
    C = c.groupby(["jeu", "cas", "methode"]).rho.median().unstack("methode").reset_index()
    return U, C


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    cur = 0.0
    for r, i in enumerate(o):
        cur = max(cur, min(1.0, (len(p) - r) * p[i]))
        adj[i] = cur
    return adj


def tests(C, meths):
    rows = []
    for j in JEUX:
        if j == "PVDAQ":
            continue
        d = C[C.jeu == j]
        ms = [m for m in meths if d[m].notna().all()]       # H-CDE : N-CMAPSS seulement
        med = d[ms].median()
        ref = med.idxmax()
        autres = [m for m in ms if m != ref]
        p = []
        for m in autres:
            x = (d[m] - d[ref]).values
            p.append(1.0 if np.allclose(x, 0) else wilcoxon(d[m], d[ref]).pvalue)
        padj = holm(p)
        rows.append(dict(jeu=j, methode=ref, mediane=med[ref], p=np.nan, p_holm=np.nan, gras=True, reference=True))
        for m, pp, pa in zip(autres, p, padj):
            rows.append(dict(jeu=j, methode=m, mediane=med[m], p=pp, p_holm=pa, gras=pa >= ALPHA, reference=False))
    return pd.DataFrame(rows)


def bootstrap(U, a, b, jeu, rng):
    d = U[U.jeu == jeu][["cas", a, b]].dropna()
    d = d.assign(x=d[a] - d[b])
    par_cas = [g.x.values for _, g in d.groupby("cas")]
    obs = np.median([np.median(x) for x in par_cas])
    stats = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, len(par_cas), len(par_cas))
        stats[i] = np.median([np.median(par_cas[k][rng.integers(0, len(par_cas[k]), len(par_cas[k]))]) for k in idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return obs, lo, hi


def contrastes(U):
    rng = np.random.default_rng(0)
    rows = []
    for j in JEUX:
        for b in REPR[1:]:
            obs, lo, hi = bootstrap(U, "lenteur", b, j, rng)
            rows.append(dict(jeu=j, contraste=f"lenteur − {b}", mediane=obs, ic_bas=lo, ic_haut=hi))
        # meilleure sans contexte contre meilleure avec contexte, par unité (biais de sélection : rapporté à titre
        # indicatif, pas comme un contraste fixé)
    return pd.DataFrame(rows)


def cd(C, meths):
    d = C[C.jeu.isin(INFORMATIFS)].dropna(subset=meths)
    R = d[meths].rank(axis=1, ascending=False)            # rang 1 = meilleure
    n, k = len(R), len(meths)
    rang_moy = R.mean().sort_values()
    chi = friedmanchisquare(*[d[m].values for m in meths])
    q = studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
    cdist = q * np.sqrt(k * (k + 1) / (6 * n))
    egal = R.assign(jeu=d.jeu.values).groupby("jeu")[meths].mean().mean().sort_values()
    return rang_moy, cdist, chi, n, egal


def figure_cd(rang_moy, cdist, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    noms = {"lenteur": "slowness", "tendance_moy": "trend, slope", "tendance_reg": "trend, regression",
            "resid_lin": "linear residual", "resid_quad": "quadratic residual", "residu_FNN": "neural residual",
            "norm_regime": "per-regime norm.", "span_W": "span$_W$", "invariant_regime": "regime-invariant SFA",
            "DANN": "DANN", "bajarunas_corr": "Bajarunas, corr.", "bajarunas_mono": "Bajarunas, mono.",
            "acp_T2": "PCA $T^2$", "acp_SPE": "PCA residual", "autoencodeur": "autoencoder", "resid_GP": "GP residual",
            "resid_GP_z": "GP residual / s.d.", "V-REx": "V-REx"}
    k = len(rang_moy)
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    lo, hi = 1, k
    ax.set_xlim(hi + 0.3, lo - 0.3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([lo, hi], [0.8, 0.8], color="k", lw=1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [0.8, 0.83], color="k", lw=1)
        ax.text(r, 0.86, str(r), ha="center", va="bottom", fontsize=8)
    items = list(rang_moy.items())
    moitie = (k + 1) // 2
    for i, (m, r) in enumerate(items):
        gauche = i < moitie
        y = 0.66 - 0.072 * (i if gauche else i - moitie)
        x_txt = lo - 0.2 if gauche else hi + 0.2
        ax.plot([r, r, x_txt], [0.8, y, y], color="k", lw=0.7)
        ax.text(x_txt, y + 0.012, f"{noms.get(m, m)} ({r:.1f})", ha="right" if gauche else "left", va="bottom",
                fontsize=7)
    # barres des groupes non distinguables
    rs = rang_moy.values
    groupes = []
    for i in range(k):
        jmax = max(j for j in range(i, k) if rs[j] - rs[i] <= cdist)
        if jmax > i and not any(a <= i and jmax <= b for a, b in groupes):
            groupes.append((i, jmax))
    for g, (i, j) in enumerate(groupes):
        y = 0.76 - 0.025 * g
        ax.plot([rs[i] - 0.03, rs[j] + 0.03], [y, y], color="k", lw=2.2)
    ax.plot([lo, lo + cdist], [0.97, 0.97], color="k", lw=1)
    ax.text(lo + cdist / 2, 0.99, f"CD = {cdist:.2f} ({n} cases)", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ICI / "figures" / f"fig5_cd.{ext}", dpi=200, bbox_inches="tight")


def main():
    U, C = charger()
    h = OUT / "vB44_par_unite.csv"
    if h.exists():                                         # H-CDE (vB44), N-CMAPSS seulement
        Lh = pd.read_csv(h)
        Lh = Lh[Lh.methode == "H-CDE"]
        ch = Lh.groupby(["cas", "graine"]).rho.median().groupby("cas").median().rename("H-CDE").reset_index()
        C = C.merge(ch.assign(jeu="N-CMAPSS"), on=["jeu", "cas"], how="left")
    v45 = OUT / "vB45_par_unite.csv"
    if v45.exists():                                       # V-REx β = 10 (vB45), jeux informatifs, régimes retenus
        Lv = pd.read_csv(v45)
        Lv = Lv[Lv.methode == "vrex_b10"]
        cv = Lv.groupby(["jeu", "cas", "graine"]).rho.median().groupby(["jeu", "cas"]).median().rename("V-REx")
        cv = cv.reset_index()
        cv.loc[cv.jeu == "C-MAPSS", "jeu"] = "C-MAPSS retenu"
        C = C.merge(cv, on=["jeu", "cas"], how="left")
    meths = [m for m in C.columns if m not in ("jeu", "cas")]
    T = tests(C, meths)
    T.to_csv(OUT / "vB40_tests.csv", index=False)
    K = contrastes(U)
    K.to_csv(OUT / "vB40_contrastes.csv", index=False)
    rang_moy, cdist, chi, n, egal = cd(C, [m for m in meths if m != "H-CDE"])   # 15 oct. : 18 méthodes
    figure_cd(rang_moy, cdist, n)
    pd.set_option("display.width", 250)
    with open(OUT / "vB40.txt", "w") as fh:
        def p(*a):
            print(*a)
            print(*a, file=fh)
        p("=== 1. Wilcoxon apparié par cas contre la meilleure du jeu, Holm (gras = p_holm ≥ 0,05) ===")
        p(T.round(3).to_string(index=False))
        p("\n=== 2. Contrastes entre représentants fixés (médiane des écarts ; IC 95 % bootstrap cas → unités) ===")
        p(K.round(3).to_string(index=False))
        p(f"\n=== 3. Différences critiques : {n} cas, Friedman χ² = {chi.statistic:.1f}, p = {chi.pvalue:.2e}, "
          f"CD (Nemenyi, α = 0,05) = {cdist:.2f} ===")
        p(rang_moy.round(2).to_string())
        p("\nrangs moyens, poids égal par jeu :")
        p(egal.round(2).to_string())
        autres = [m for m in meths if m not in METH12]
        if autres:
            p("\n=== méthodes ajoutées (R10, R12) : médiane par jeu ===")
            p(C.groupby("jeu")[autres].median().reindex(JEUX).round(2).to_string())


if __name__ == "__main__":
    main()
