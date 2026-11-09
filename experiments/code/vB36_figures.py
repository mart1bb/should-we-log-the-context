"""vB36 — Figures 1–4 du papier ICDE (G2). Aucun calcul nouveau : relit les résultats versionnés.

Fig. 1 : H-CDE face à une classe de vol jamais vue (it17 : AUROC référence contre non vue ; réponse relative).
Fig. 2 : retirer le contexte par un modèle (résidu quadratique) contre la lenteur, par cas, un panneau par jeu (vB35).
Fig. 3 : un représentant fixé d'avance par famille (lenteur ; résidu quadratique ; `span_W` ; Bajarunas corr),
         médiane et écart interquartile par jeu (vB35) — pas de « meilleure de la famille » (biais de sélection).
Fig. 4 : quand le contexte devient nécessaire — un panneau par condition cassée : dérive d'usage (N-CMAPSS, vB37,
         avec le contrôle de même taille sans dérive),
         autre processus lent (PVDAQ, vB35), signal lent faible (batteries, vB35), signature dépendante du régime
         (C-MAPSS FD002 R20, vB35).
Couleurs : palette catégorielle de référence (dataviz), validée (4 emplacements, mode clair) ; hachures en plus pour
l'impression en niveaux de gris. Sortie : voie_B_avec_exo/figures/fig{1..4}.pdf/.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

VOIE = Path(__file__).resolve().parents[1]
RES, TRONC = VOIE / "results", VOIE.parent / "results"
FIG = VOIE / "figures"
ENCRE, ENCRE2, GRILLE = "#0b0b0b", "#52514e", "#e4e3df"
REP = {  # représentant par famille : (libellé, couleur, hachure)
    "lenteur": ("Context-free: slowness (SFA)", "#2a78d6", ""),
    "resid_quad": ("Context removal: quadratic residual", "#eb6834", "////"),
    "span_W": ("Context invariance: span$_W$", "#1baf7a", "...."),
    "bajarunas_corr": ("Latent with context: Bajarunas et al.", "#4a3aa7", "xxxx"),
}
JEUX = ["N-CMAPSS", "C-MAPSS", "batteries", "roulements", "gaz", "PVDAQ"]
NOMS = {"N-CMAPSS": "N-CMAPSS", "C-MAPSS": "C-MAPSS withheld", "batteries": "Batteries", "roulements": "Bearings",
        "gaz": "Gas sensors", "PVDAQ": "PV systems"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": ENCRE2, "axes.labelcolor": ENCRE,
                     "xtick.color": ENCRE2, "ytick.color": ENCRE2, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def axe_propre(ax, y0=None):
    ax.grid(axis="y", color=GRILLE, lw=0.6)
    ax.set_axisbelow(True)
    if y0 is not None:
        ax.axhline(y0, color=ENCRE2, lw=0.8)


def sauver(fig, nom):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{nom}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{nom}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1():
    """15 oct. : refaite depuis vB44 (12 cas du banc, 3 graines, tous les runs, orientation sans étiquettes), en
    Spearman comme le reste du papier ; vu = unité de référence (classe Fc1, hors entraînement), non vu = Fc2/Fc3.
    Par cas : médiane sur les graines (et sur les unités non vues). L'ancienne version (it17, AUROC, porte D-16)
    reste dans l'historique git."""
    L = pd.read_csv(RES / "vB44_par_unite_avec_ref.csv")
    L = L[L.methode == "H-CDE"]
    ref = L[L.role == "ref"].set_index(["cas", "graine"])
    nv = L[L.role == "non_vue"].copy()
    nv["rel"] = nv.reponse / ref.reponse.reindex(pd.MultiIndex.from_frame(nv[["cas", "graine"]])).values
    d = pd.DataFrame({
        "vu": ref.rho.groupby("cas").median(),
        "nv": nv.groupby(["cas", "graine"]).rho.median().groupby("cas").median(),
        "rel": nv.groupby("cas").rel.median()})
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.9), gridspec_kw={"width_ratios": [1.3, 1]})
    for _, r in d.iterrows():
        a.plot([0, 1], [r.vu, r.nv], color=ENCRE2, lw=0.6, alpha=0.6)
    a.scatter(np.zeros(len(d)), d.vu, s=10, color="#2a78d6", zorder=3)
    a.scatter(np.ones(len(d)), d.nv, s=10, color="#2a78d6", zorder=3)
    a.set_xticks([0, 1], ["seen flight\nclass", "unseen flight\nclass"])
    a.set_xlim(-0.3, 1.3)
    a.set_ylabel("Spearman with degradation")
    a.set_title(f"(a) ranking, {len(d)} cases", loc="left")
    axe_propre(a)
    a.text(-0.2, d[["vu", "nv"]].min().min(), f"lower in {int((d.nv < d.vu).sum())}/{len(d)}", ha="left",
           color=ENCRE2, fontsize=7)
    x = np.random.default_rng(0).uniform(-0.15, 0.15, len(d))
    bas = -0.5                                          # cas dont la réponse de référence est quasi nulle : annotés
    hors_axe = d.rel < bas
    b.scatter(x[~hors_axe], d.rel[~hors_axe], s=10, color="#2a78d6")
    for xi, r in zip(x[hors_axe], d.rel[hors_axe]):
        b.annotate(f"{r:.1f}", (xi, bas + 0.02), xytext=(xi, bas + 0.35), fontsize=6.5, color=ENCRE2, ha="center",
                   arrowprops=dict(arrowstyle="->", color=ENCRE2, lw=0.6))
    b.set_ylim(bas, 1.5)
    b.axhline(1, color=ENCRE2, lw=0.8, ls="--")
    b.text(0.45, 1.02, "same response", color=ENCRE2, fontsize=7, va="bottom", ha="right")
    b.set_xticks([])
    b.set_xlim(-0.5, 0.5)
    b.set_ylabel("score change per unit of\ndegradation, unseen / seen")
    b.set_title("(b) response to degradation", loc="left")
    axe_propre(b)
    fig.tight_layout()
    sauver(fig, "fig1_hcde_unseen_class")
    print("Fig. 1 :", d.median().round(2).to_dict(), "baisse dans", int((d.nv < d.vu).sum()), "cas sur", len(d))


def table_retenus():
    """vB35 avec, pour C-MAPSS, les seuls cas à régime retenu (hors support), comme le tableau III (14 oct.)."""
    t = pd.read_csv(RES / "vB35_table_unifiee.csv")
    return t[(t.jeu != "C-MAPSS") | t.cas.str.contains(r" R\d")]


def fig2():
    t = table_retenus()
    fig, axs = plt.subplots(1, 6, figsize=(7.16, 1.75), sharex=True, sharey=True)
    for ax, j in zip(axs, JEUX):
        x = t[t.jeu == j]
        ax.plot([-1, 1], [-1, 1], color=ENCRE2, lw=0.6, ls="--")
        ax.scatter(x.lenteur, x.resid_quad, s=9, color="#eb6834", edgecolor="white", lw=0.3, zorder=3)
        bas = (x.resid_quad < x.lenteur - 0.05).mean()
        ax.set_title(f"{NOMS[j]}\nbelow diagonal: {bas:.0%}", loc="left", fontsize=6.5)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.grid(color=GRILLE, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_aspect("equal")
    axs[0].set_ylabel("context removed by a model\n(quadratic residual + SFA)")
    fig.supxlabel("context-free slowness (SFA)", fontsize=8, y=0.02)
    fig.tight_layout(w_pad=0.9)
    sauver(fig, "fig2_removal_vs_slowness")


def barres(ax, donnees, methodes, largeur=0.2):
    """donnees : dict jeu → DataFrame ; médiane (barre) et écart interquartile (trait)."""
    n = len(methodes)
    for i, m in enumerate(methodes):
        lab, coul, hach = REP[m]
        for k, (j, x) in enumerate(donnees.items()):
            v = x[m].dropna()
            pos = k + (i - (n - 1) / 2) * largeur
            ax.bar(pos, v.median(), width=largeur * 0.88, color=coul, hatch=hach, edgecolor="white", lw=0.4,
                   label=lab if k == 0 else None)
            ax.vlines(pos, v.quantile(0.25), v.quantile(0.75), color=ENCRE, lw=0.8)


def fig3():
    t = table_retenus()
    fig, ax = plt.subplots(figsize=(7.16, 2.2))
    donnees = {j: t[t.jeu == j] for j in JEUX}
    barres(ax, donnees, list(REP))
    ax.set_xticks(range(len(JEUX)), [f"{NOMS[j]}\n({len(donnees[j])} cases)" for j in JEUX])
    ax.axvspan(len(JEUX) - 1.5, len(JEUX) - 0.5, color=GRILLE, alpha=0.5, lw=0, zorder=0)
    ax.text(len(JEUX) - 1, 0.95, "uninformative:\nno method tracks age", ha="center", va="top", fontsize=6.5,
            color=ENCRE2)
    ax.set_ylabel("Spearman with ground truth\n(median, IQR over cases)")
    ax.set_ylim(-0.75, 1.0)
    axe_propre(ax, 0)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.6)
    fig.tight_layout()
    sauver(fig, "fig3_families")


def fig4():
    t = pd.read_csv(RES / "vB35_table_unifiee.csv")
    r37 = pd.read_csv(RES / "vB37_derive_span_w.csv")
    derive = r37[r37.scenario == "montante_h0.30"]
    controle = r37[r37.scenario == "controle montante_h0.30"].lenteur.median()
    panneaux = [
        ("C1 violated: usage drifts slowly\n(N-CMAPSS, generated drift)", derive),
        ("C2 violated: other slow processes\n(PV systems)", t[t.jeu == "PVDAQ"]),
        ("C3 violated: slow direction changes\nwith usage (batteries)", t[t.jeu == "batteries"]),
        ("C3 violated: slow direction changes\nwith regime (C-MAPSS FD002, 20 kft unseen)",
         t[t.cas.str.startswith("FD002 R20")]),
    ]
    meth = ["lenteur", "resid_quad", "span_W"]
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.0), sharey=True)
    for ax, (titre, x) in zip(axs, panneaux):
        for i, m in enumerate(meth):
            lab, coul, hach = REP[m]
            v = x[m].dropna()
            ax.bar(i, v.median(), width=0.7, color=coul, hatch=hach, edgecolor="white", lw=0.4,
                   label=lab if ax is axs[0] else None)
            ax.scatter(i + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), v, s=5, color=ENCRE, zorder=3)
        if x is derive:                           # lenteur au contrôle de même taille, sans dérive
            ax.hlines(controle, -0.4, 0.4, color=ENCRE, lw=1.0, ls="--")
            ax.text(0.45, controle, "same size,\nno drift", fontsize=6, va="center", color=ENCRE2)
        ax.set_xticks([])
        ax.set_title(titre, loc="left", fontsize=6.5)
        ax.set_ylim(-1, 1)
        axe_propre(ax, 0)
    axs[0].set_ylabel("Spearman with ground truth")
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.6)
    fig.tight_layout()
    sauver(fig, "fig4_conditions")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("figures écrites dans", FIG)
