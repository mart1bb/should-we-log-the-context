"""vB68 — Tableaux et figure de la partie 2 du papier (journal d'événements), depuis vB67, vB62 (placebo) et vB64.

Valeurs : dégât = sans journal − propre ; gain de chaque contenu = meilleure réponse avec ce contenu − sans correction
(un opérateur peut toujours ne pas corriger : c'est la référence ; correction de lecture du 19 oct., les écarts
« catégorie contre date » de vB67 mesuraient surtout la nuisance de la correction par la date seule).
Résumé d'un cas (fixé avant de produire les tableaux, pour ne pas choisir la méthode après coup) : médiane, sur les
quatre méthodes (lenteur, tendance_reg, resid_quad, span_W), de la quantité du cas (elle-même moyenne des 2 tirages).
Par jeu et catégorie : médiane sur les cas, IC à 95 % par bootstrap sur les cas (2 000 tirages, graine 0), Wilcoxon
apparié, Holm sur toutes les cases du tableau.
CARE : AUROC par parc (jeux avec panne contre jeux normaux, score journalier maximal / seuil), IC par bootstrap
stratifié sur les jeux (2 000 tirages, graine 0).
Sorties : papier/tab_events.tex, papier/tab_care.tex, papier/fig6_event_logs.pdf, results/vB68.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

RACINE = Path(__file__).resolve().parent.parent
RES, PAP = RACINE / "results", RACINE / "papier"
METH = ["lenteur", "tendance_reg", "resid_quad", "span_W"]
JEUX = [("C-MAPSS", "C-MAPSS"), ("N-CMAPSS", "N-CMAPSS"), ("batteries", "Batteries"), ("roulements", "Bearings")]
CATS = [("mesure", "measurement"), ("production", "operation"), ("maintenance", "maintenance")]
QUANT = [("degat", "damage"), ("g_detection", "detected"), ("g_date", "date"), ("g_cat", "+ category"),
         ("g_capteurs", "+ sensors")]


def valeurs_par_cas():
    R = pd.read_csv(RES / "vB67_cas.csv")
    L = R.melt(id_vars=["jeu", "cas", "ampleur", "categorie", "tirage", "reponse", "estimation"], value_vars=METH,
               var_name="methode", value_name="rho")
    propre = L[L.reponse == "propre"].set_index(["jeu", "cas", "methode"]).rho
    L = L[L.reponse != "propre"].copy()
    L["rep"] = L.reponse + "|" + L.estimation
    P = L.pivot_table(index=["jeu", "cas", "ampleur", "categorie", "tirage", "methode"], columns="rep", values="rho")
    P = P.groupby(level=["jeu", "cas", "ampleur", "categorie", "methode"]).mean()
    best = lambda n: P[[f"{n}|est_brut", f"{n}|est_ctx"]].max(axis=1)                  # noqa: E731
    V = pd.DataFrame(index=P.index)
    V["degat"] = P["sans_journal|-"] - propre.reindex(P.index.droplevel(["ampleur", "categorie"])).values
    V["g_detection"] = P["detection|est_brut"] - P["sans_journal|-"]
    V["g_date"] = best("date") - P["sans_journal|-"]
    V["g_cat"] = best("cat") - P["sans_journal|-"]
    V["g_capteurs"] = best("cat_capteurs") - P["sans_journal|-"]
    return V.groupby(level=["jeu", "cas", "ampleur", "categorie"]).median()          # médiane sur les méthodes


def resume(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    bs = np.median(x[np.random.default_rng(0).integers(0, len(x), (2000, len(x)))], axis=1)
    p = wilcoxon(x).pvalue if len(x) >= 3 and np.any(x != 0) else np.nan
    return np.median(x), np.percentile(bs, 2.5), np.percentile(bs, 97.5), p, len(x)


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    idx = ok[np.argsort(p[ok])]
    cur = 0.0
    for r, i in enumerate(idx):
        cur = max(cur, min(1.0, (len(idx) - r) * p[i]))
        out[i] = cur
    return out


def fmt(v):
    return ("$-$" if v < -0.005 else "") + f"{abs(v):.2f}"


def tableau_evenements(V, lignes):
    S = []
    for j, _ in JEUX:
        for c, _ in CATS:
            for q, _ in QUANT:
                x = V.xs((j, 1.0, c), level=["jeu", "ampleur", "categorie"])[q].values
                if q == "g_cat" and c == "mesure":                    # identique à la date (même réponse)
                    S.append(dict(jeu=j, cat=c, q=q, med=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=len(x)))
                    continue
                m, lo, hi, p, n = resume(x)
                S.append(dict(jeu=j, cat=c, q=q, med=m, lo=lo, hi=hi, p=p, n=n))
    S = pd.DataFrame(S)
    S["p_holm"] = holm(S.p.values)
    out = ["% Généré par code/vB68_partie2_papier.py — ne pas éditer à la main.",
           r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
           r"Dataset & Change & Damage & \multicolumn{4}{c}{Gain over no correction, with} \\",
           r"\cmidrule(l){4-7}",
           r" & & & detected & logged & + category & + sensors \\",
           r" & & & dates & dates & & \\", r"\midrule"]
    for j, jn in JEUX:
        n = int(S[(S.jeu == j)].n.iloc[0])
        for i, (c, cn) in enumerate(CATS):
            cells = []
            for q, _ in QUANT:
                r = S[(S.jeu == j) & (S.cat == c) & (S.q == q)].iloc[0]
                if np.isnan(r.med):
                    cells.append("--")
                else:
                    t = fmt(r.med)
                    cells.append(rf"\textbf{{{t}}}" if r.p_holm < 0.05 else t)
            tete = f"{jn} ({n})" if i == 0 else ""
            out.append(f"{tete} & {cn} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule" if j != JEUX[-1][0] else r"\bottomrule")
    out.append(r"\end{tabular}")
    (RES / "vB68_tab_events_vB67.tex").write_text("\n".join(out) + "\n")   # Tableau IV du papier : vB74 (20 oct.)
    lignes.append("Tableau des événements (f = 1, médiane des méthodes ; IC, p de Holm)\n"
                  + S.round(3).to_string(index=False))
    return S


def figure(V, lignes):
    couleurs = {"C-MAPSS": "#0072B2", "N-CMAPSS": "#D55E00", "batteries": "#009E73", "roulements": "#7F7F7F"}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), sharex=True, sharey=True)
    panneaux = [("degat", "production", "(a) damage"),
                ("g_date", "production", "(b) gain, logged dates"),
                ("g_capteurs", "production", "(c) gain, + category + sensors")]
    for a, (q, c, titre) in zip(ax, panneaux):
        for j, jn in JEUX:
            pts = [resume(V.xs((j, f, c), level=["jeu", "ampleur", "categorie"])[q].values) for f in (0.25, 1.0, 2.0)]
            m = [p[0] for p in pts]
            a.plot([0.25, 1, 2], m, marker="o", ms=3, lw=1.5, color=couleurs[j], label=jn)
            if j in ("C-MAPSS", "N-CMAPSS"):                  # IC lisibles seulement avec 30 et 12 cas
                a.fill_between([0.25, 1, 2], [p[1] for p in pts], [p[2] for p in pts], color=couleurs[j], alpha=0.15,
                               lw=0)
            lignes.append(f"figure {q} {c} {j} : " + " ; ".join(f"f={f} {p[0]:.2f} [{p[1]:.2f}, {p[2]:.2f}]"
                                                                  for f, p in zip((0.25, 1, 2), pts)))
        a.axhline(0, color="#999999", lw=0.6)
        a.set_title(titre, fontsize=8, loc="left")
        a.set_xlabel("magnitude $f$", fontsize=8)
        a.tick_params(labelsize=7)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("Spearman difference", fontsize=8)
    ax[1].legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(PAP / "fig6_event_logs.pdf")


def tableau_care(lignes):
    R = pd.read_csv(RES / "vB64_jeux.csv")
    rng = np.random.default_rng(0)
    out_rows = []
    for (sc, te, inf), g in R.groupby(["scoreur", "entrainement", "inference"]):
        row = dict(scoreur=sc, entrainement=te, inference=inf)
        for parc, h in g.groupby("parc"):
            h = h.dropna(subset=["max_rapport"])
            y, s = (h.label == "anomaly").values, h.max_rapport.values
            auc = roc_auc_score(y, s)
            ia, inn = np.where(y)[0], np.where(~y)[0]
            bs = [roc_auc_score(np.r_[np.ones(len(ia)), np.zeros(len(inn))],
                                np.r_[s[rng.choice(ia, len(ia))], s[rng.choice(inn, len(inn))]]) for _ in range(2000)]
            row[parc] = (auc, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(h))
        out_rows.append(row)
    noms = {("sans_ctx", "tout", "tout"): ("no context", "all", "all"),
            ("sans_ctx", "journal", "tout"): ("no context", "normal", "all"),
            ("sans_ctx", "tout", "journal"): ("no context", "all", "normal"),
            ("ctx", "tout", "tout"): ("context", "all", "all"),
            ("ctx", "journal", "tout"): ("context", "normal", "all"),
            ("ctx", "tout", "journal"): ("context", "all", "normal")}
    T = ["% Généré par code/vB68_partie2_papier.py — ne pas éditer à la main.",
         r"\begin{tabular}{@{}lllccc@{}}", r"\toprule",
         r"Model & Training rows & Scored rows & Farm A & Farm B & Farm C \\", r"\midrule"]
    for k, (a, b, c) in noms.items():
        r = next(x for x in out_rows if (x["scoreur"], x["entrainement"], x["inference"]) == k)
        cells = [f"{r[p][0]:.2f} [{r[p][1]:.2f}, {r[p][2]:.2f}]" for p in "ABC"]
        T.append(f"{a} & {b} & {c} & " + " & ".join(cells) + r" \\")
        lignes.append(f"CARE {k} : " + " ; ".join(f"{p} {r[p][0]:.3f} [{r[p][1]:.3f}, {r[p][2]:.3f}] n={r[p][3]}"
                                                  for p in "ABC"))
    T += [r"\bottomrule", r"\end{tabular}"]
    (PAP / "tab_care.tex").write_text("\n".join(T) + "\n")


def placebo(lignes):
    R = pd.read_csv(RES / "vB62_cas.csv")
    pr = R[R.reponse == "propre"].set_index(["jeu", "cas"])[METH]
    P = R[(R["mode"] == "placebo") & (R.estimation == "est_brut") & (R.reponse == "date")]
    d = (P.set_index(["jeu", "cas"])[METH] - pr.reindex(P.set_index(["jeu", "cas"]).index).values).median(axis=1)
    lignes.append("Placebo vB62 (date seule, estimation sur la vie, médiane des méthodes puis des cas et tirages) : "
                  + str(d.groupby(level="jeu").median().round(3).to_dict()))


def alarmes(lignes):
    A = pd.read_csv(RES / "vB67_alarmes.csv")
    A = A[(A.jeu == "C-MAPSS")]
    t = A.groupby(["categorie", "ampleur", "condition"])[["faux_par_unite", "detection", "part_expliquee", "masquage"]]\
        .median().round(3)
    lignes.append("Alarmes C-MAPSS (médianes)\n" + t.to_string())


def main():
    lignes = []
    V = valeurs_par_cas()
    tableau_evenements(V, lignes)
    figure(V, lignes)
    tableau_care(lignes)
    placebo(lignes)
    alarmes(lignes)
    (RES / "vB68.txt").write_text("\n\n".join(lignes) + "\n")
    print("\n\n".join(lignes))


if __name__ == "__main__":
    main()
