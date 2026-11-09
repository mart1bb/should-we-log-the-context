"""vB38 — Tableau III du papier généré depuis les résultats (R2, R9, R10, R12, R13 de PAPIER_ICDE § 8).

Aucune nouvelle mesure. Source : results/vB39_cas.csv (sinon vB35_table_unifiee.csv), H-CDE : vB44_par_unite.csv
(N-CMAPSS seulement) s'il existe. C-MAPSS est séparé en deux colonnes, régime retenu (`R*`, hors support, 30 cas) et
contrôle (`C0_*`, régime vu, 30 cas), comme demandé par la relecture (W5).
Gras : si results/vB40_tests.csv existe, méthode non significativement différente de la meilleure du jeu (Wilcoxon
apparié par cas, Holm, vB40) ; sinon, provisoire : à moins de 0,05 du meilleur de la colonne.
Sortie : papier/tab_main.tex (\\input dans sections/05_results.tex).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ICI = Path(__file__).resolve().parent.parent
RES = ICI / "results"
LIGNES = [
    ("Context-free, temporal prior", [("lenteur", "slowness (SFA)"), ("tendance_moy", "trend, mean slope"),
                                      ("tendance_reg", "trend, regression on time")]),
    ("Context-free, no temporal prior", [("acp_T2", "PCA, Hotelling $T^2$"), ("acp_SPE", "PCA, residual (SPE)"),
                                         ("autoencodeur", "autoencoder")]),
    ("Context removal", [("resid_lin", "linear residual + SFA"), ("resid_quad", "quadratic residual + SFA"),
                         ("residu_FNN", "neural residual (FNN)"), ("resid_GP", "GP residual"),
                         ("resid_GP_z", "GP residual / predictive s.d."),
                         ("norm_regime", "per-regime normalization + SFA")]),
    ("Context invariance", [("span_W", "span$_W$"), ("invariant_regime", "regime-invariant SFA (linear)"),
                            ("DANN", "DANN (non-linear)"), ("V-REx", "V-REx with slowness")]),
    ("Latent with context", [("bajarunas_corr", "Bajarunas et al., correlation"),
                             ("bajarunas_mono", "Bajarunas et al., monotonicity"), ("H-CDE", "H-CDE")]),
]
COLS = ["N-CMAPSS", "C-MAPSS retenu", "C-MAPSS contrôle", "batteries", "roulements", "gaz", "PVDAQ"]


def charger():
    src = RES / "vB39_cas.csv"
    t = pd.read_csv(src if src.exists() else RES / "vB35_table_unifiee.csv")
    cm = t.jeu == "C-MAPSS"
    t.loc[cm & t.cas.str.contains(r" R\d"), "jeu"] = "C-MAPSS retenu"
    t.loc[cm & t.cas.str.contains("C0_"), "jeu"] = "C-MAPSS contrôle"
    h = RES / "vB44_par_unite.csv"
    if h.exists():
        L = pd.read_csv(h)
        L = L[L.methode == "H-CDE"]
        c = L.groupby(["cas", "graine"]).rho.median().groupby("cas").median().rename("H-CDE").reset_index()
        t = t.merge(c.assign(jeu="N-CMAPSS"), on=["jeu", "cas"], how="left")
    v45 = RES / "vB45_par_unite.csv"
    if v45.exists():
        L = pd.read_csv(v45)
        L = L[L.methode == "vrex_b10"]
        c = L.groupby(["jeu", "cas", "graine"]).rho.median().groupby(["jeu", "cas"]).median().rename("V-REx").reset_index()
        c.loc[c.jeu == "C-MAPSS", "jeu"] = "C-MAPSS retenu"
        t = t.merge(c, on=["jeu", "cas"], how="left")
    return t


def fmt(x):
    return "--" if pd.isna(x) else f"{x:.2f}".replace("-", "$-$")


def main():
    t = charger()
    lignes = [(fam, [(m, n) for m, n in l if m in t.columns]) for fam, l in LIGNES]
    lignes = [(f, l) for f, l in lignes if l]
    ms = [m for _, l in lignes for m, _ in l]
    med = t.groupby("jeu")[ms].median().reindex(COLS)
    n = t.groupby("jeu").size().reindex(COLS)
    tests = RES / "vB40_tests.csv"
    if tests.exists():
        T = pd.read_csv(tests)
        gras = {(r.jeu, r.methode): bool(r.gras) for r in T.itertuples()}
        regle = "tests"
    else:
        best = med.max(axis=1)
        gras = {(c, m): med.loc[c, m] >= best[c] - 0.05 for c in COLS for m in ms if not np.isnan(med.loc[c, m])}
        regle = "0,05"
    out = []
    for fam, l in lignes:
        for i, (m, nom) in enumerate(l):
            cell = []
            for c in COLS:
                s = fmt(med.loc[c, m])
                cell.append(f"\\textbf{{{s}}}" if c != "PVDAQ" and gras.get((c, m), False) else s)
            lib = fam.replace(", ", ",\\\\ ")
            tete = f"\\multirow{{{len(l)}}}{{*}}{{\\shortstack[l]{{{lib}}}}}" if i == 0 else ""
            out.append(f"{tete} & {nom} & " + " & ".join(cell) + " \\\\")
        out.append("\\midrule")
    out[-1] = "\\bottomrule"
    entete = ("\\toprule\nFamily & Method & N-CMAPSS & \\multicolumn{2}{c}{C-MAPSS} & Batteries & Bearings & "
              "Gas sensors & PV systems \\\\\n & & & withheld & control & & & & \\\\\n & & "
              + " & ".join(f"({k})" for k in n) + " \\\\\n\\midrule")
    tex = (f"% Généré par code/vB38_tables_papier.py (gras : {regle}) — ne pas éditer à la main.\n"
           "\\begin{tabular}{@{}llrrrrrrr@{}}\n" + entete + "\n" + "\n".join(out) + "\n\\end{tabular}\n")
    (ICI / "papier" / "tab_main.tex").write_text(tex)
    print(f"gras : {regle}")
    print(med.T.round(2).to_string())


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------------------------------
# 15 oct. : tableau « dans le support / hors support » (2e relecture, W1), depuis vB47 et vB39.
SUPPORT = [("lenteur", "slowness (SFA)"), ("tendance_reg", "trend, regression on time"),
           ("resid_lin", "linear residual"), ("resid_quad", "quadratic residual"), ("residu_FNN", "neural residual"),
           ("resid_GP", "GP residual"), ("span_W", "span$_W$")]
JEUX_S = [("N-CMAPSS", "N-CMAPSS"), ("C-MAPSS", "C-MAPSS"), ("batteries", "Batteries"), ("roulements", "Bearings"),
          ("gaz", "Gas sensors")]


def tab_support():
    dans = pd.read_csv(RES / "vB47_cas.csv")
    hors = pd.read_csv(RES / "vB39_cas.csv")
    cm = hors.jeu == "C-MAPSS"
    dans = pd.concat([dans, hors[cm & hors.cas.str.contains("C0_")]], ignore_index=True)
    hors = hors[~cm | hors.cas.str.contains(r" R\d")]
    ms = [m for m, _ in SUPPORT]
    md = dans.groupby("jeu")[ms].median()
    mh = hors.groupby("jeu")[ms].median()
    nd, nh = dans.groupby("jeu").size(), hors.groupby("jeu").size()
    tete = " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in JEUX_S)
    sous = " & ".join("in & out" for _ in JEUX_S)
    nb = " & ".join(f"({nd[j]}) & ({nh[j]})" for j, _ in JEUX_S)
    lignes = [f"{nom} & " + " & ".join(f"{fmt(md.loc[j, m])} & {fmt(mh.loc[j, m])}" for j, _ in JEUX_S) + " \\\\"
              for m, nom in SUPPORT]
    tex = ("% Généré par code/vB38_tables_papier.py (tab_support) — ne pas éditer à la main.\n"
           "\\begin{tabular}{@{}l" + "rr" * len(JEUX_S) + "@{}}\n\\toprule\nMethod & " + tete + " \\\\\n & " + sous
           + " \\\\\n & " + nb + " \\\\\n\\midrule\n" + "\n".join(lignes) + "\n\\bottomrule\n\\end{tabular}\n")
    (ICI / "papier" / "tab_support.tex").write_text(tex)
    print(pd.concat({"in": md.T, "out": mh.T}, axis=1).round(2).to_string())


if __name__ == "__main__":
    tab_support()
