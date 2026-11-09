"""vB22 — Batteries NASA à charges aléatoires (Randomized Battery Usage, PCoE) : troisième jeu du papier B.

Données : `../../../Datasets/NASA_Batteries_RW/` (téléchargées le 25 sept. avec l'accord de Martin ;
zenodo 15277374, CC BY 4.0 ; citer Bole, Kulkarni, Daigle, PHM 2014). Cellules 18650 ; décharge
« random walk » : une nouvelle consigne de courant chaque minute, tirée d'une loi propre au groupe ; tous
les ~50 cycles, une décharge de référence à 1 A mesure la capacité (vérité de santé, meilleure que l'âge).

Structure, déclarée avant mesure (analogie avec N-CMAPSS) :
  - unité = batterie ; « vol » = un cycle RW (décharge random walk entre deux recharges) ; fenêtre = une
    étape RW d'une minute à courant constant ;
  - capteurs par étape S : tension moyenne, tension de début, de fin, variation de tension sur l'étape,
    température moyenne, variation de température (températures sentinelles < −100 °C → étape écartée) ;
  - variables d'opération W : courant de consigne (médiane de l'étape) et charge déjà débitée depuis la
    dernière recharge (Ah, proxy de l'état de charge) — elles changent d'une étape à l'autre : variation
    RAPIDE des conditions, à l'intérieur de chaque batterie ;
  - vérité : capacité de référence, interpolée linéairement entre les décharges de référence, par cycle.
Groupes : entraînement = uniformes à température ambiante (RW1–8 ; courants 0,5–4,1 A) ; test = biaisées
basses RW13–16 et hautes RW17–20 à température ambiante (courants jusqu'à 5,1 A : hors support ; loi
déplacée), et biaisées basses RW21–24 et hautes RW25–28 à 40 °C (en plus, température ambiante jamais vue,
NON mesurée comme W). Contrôle : une batterie uniforme laissée de côté, entraînement sur les 7 autres.
RW9–12 exclues (protocole différent, sans cycles RW).
Méthodes (même SFA en aval ; W n'entre que par le modèle ou par `span_W`) :
  - `sfa_brut` : SFA commune sur S (méthode de la voie A, repère cité) ;
  - `sfa_resid_lin`, `sfa_resid_quad` : S − modèle de condition en W (linéaire ; W, W², produits) ;
  - `span_W` : version 2 (DB-08) ;
  - `sonde` : ridge supervisée (S, W) → capacité, apprise sur l'entraînement (plafond, DB-01).
Mesures par batterie de test : Spearman(score du cycle, capacité) (orienté : baisse de capacité =
dégradation) ; aussi Spearman(score, n° de cycle). Diagnostics sans étiquettes : R̂, D (SFA brute).

Lecture fixée AVANT de mesurer (médianes par groupe) :
  - revendication 2 reformulée (JOURNAL, vB19) : ici les conditions varient DANS l'unité → retirer les
    conditions par un modèle doit dégrader le CLASSEMENT hors support : `sfa_resid_*` ≥ 0,1 sous `sfa_brut`
    sur les groupes biaisés, à ± 0,05 sur le contrôle ;
  - `span_W` à ± 0,05 de `sfa_brut` (pas de dérive d'usage dans une batterie : la carte dit que W ne sert
    pas ici) ;
  - D bas (< 0,2) ; R̂ rapporté.
Si la SFA brute elle-même échoue (Spearman < 0,5 sur le contrôle), le jeu n'est pas informatif pour cette
question avec ces caractéristiques ; on le dit sans changer de caractéristiques après coup.

Machine : Mac. Cache (non versionné) : voie_B_avec_exo/cache/batteries_etapes.pkl.
Sortie : voie_B_avec_exo/results/vB22_*.csv, vB22.txt
"""
from __future__ import annotations

import glob
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import vB01_sonde_marge as v  # noqa: E402  (ajoute aussi ../code au chemin)
import vB08_espace_conditions as v8  # noqa: E402
import vB14_r_sans_etiquettes as r14  # noqa: E402
import vB15_detecteur_piege as d15  # noqa: E402
import sfa  # noqa: E402

import os
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ICI.parents[1] / "data"))  # données publiques, voir README
RACINE = DATA_ROOT / "NASA_Batteries_RW"
CACHE = ICI.parent / "cache/batteries_etapes.pkl"
S_B = ["V_moy", "V_debut", "V_fin", "dV", "T_moy", "dT"]
W_B = ["I", "Ah"]
GROUPES = {"uniforme": [1, 2, 3, 4, 5, 6, 7, 8], "biaisee_basse_TA": [13, 14, 15, 16],
           "biaisee_haute_TA": [17, 18, 19, 20], "biaisee_basse_40C": [21, 22, 23, 24],
           "biaisee_haute_40C": [25, 26, 27, 28]}


def arr(x):
    return np.atleast_1d(np.asarray(x, dtype=float))


def lire_batterie(chemin: str) -> pd.DataFrame:
    bat = int(re.search(r"RW(\d+)\.mat", chemin).group(1))
    st = sio.loadmat(chemin, simplify_cells=True)["data"]["step"]
    lignes, refs, cycle, ah = [], [], 0, 0.0
    for s in st:
        com, typ = s["comment"], s["type"]
        if com == "charge (after random walk discharge)" and typ == "C":
            cycle, ah = cycle + 1, 0.0
            continue
        if com == "reference discharge":
            refs.append((cycle, float(np.trapezoid(arr(s["current"]), arr(s["relativeTime"])) / 3600)))
            continue
        if com != "discharge (random walk)" or typ != "D":
            continue
        t, V, I, T = arr(s["relativeTime"]), arr(s["voltage"]), arr(s["current"]), arr(s["temperature"])
        if len(t) < 5 or (T < -100).any():
            ah += float(np.trapezoid(I, t) / 3600) if len(t) > 1 else 0.0
            continue
        lignes.append(dict(bat=bat, cycle=cycle, I=float(np.median(I)), Ah=ah, V_moy=V.mean(), V_debut=V[0],
                           V_fin=V[-1], dV=V[-1] - V[0], T_moy=T.mean(), dT=T[-1] - T[0]))
        ah += float(np.trapezoid(I, t) / 3600)
    e = pd.DataFrame(lignes)
    r = pd.DataFrame(refs, columns=["cycle", "capacite"]).groupby("cycle").capacite.mean()
    e["capacite"] = np.interp(e.cycle, r.index.values, r.values)
    return e


def charger() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    fichiers = [f for f in glob.glob(str(RACINE / "*/*/data/Matlab/RW*.mat"))
                if int(re.search(r"RW(\d+)\.mat", f).group(1)) in sum(GROUPES.values(), [])]
    df = pd.concat([lire_batterie(f) for f in sorted(fichiers)], ignore_index=True)
    CACHE.parent.mkdir(exist_ok=True)
    df.to_pickle(CACHE)
    return df


def phi(W):
    prod = np.stack([W[:, i] * W[:, j] for i, j in combinations(range(W.shape[1]), 2)], 1)
    return np.hstack([W, W ** 2, prod])


def scores(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    Str, Ste = tr[S_B].values, te[S_B].values
    muW, sdW = tr[W_B].values.mean(0), tr[W_B].values.std(0)
    Wtr, Wte = (tr[W_B].values - muW) / sdW, (te[W_B].values - muW) / sdW
    u, c = tr.bat.values, tr.cycle.values
    out = {"sfa_brut": sfa.ajuster(Str, u, c).score(Ste)}
    f = LinearRegression().fit(Wtr, Str)
    out["sfa_resid_lin"] = sfa.ajuster(Str - f.predict(Wtr), u, c).score(Ste - f.predict(Wte))
    q = LinearRegression().fit(phi(Wtr), Str)
    out["sfa_resid_quad"] = sfa.ajuster(Str - q.predict(phi(Wtr)), u, c).score(Ste - q.predict(phi(Wte)))
    out["span_W"] = v8.span_W(Str, tr[W_B].values, u, c, reduire=False)[0].score(Ste)
    X = np.hstack([Str, Wtr])
    out["sonde"] = Ridge(alpha=1.0).fit(X, -tr.capacite.values).predict(np.hstack([Ste, Wte]))
    return out


def mesurer(tr, te, cadre):
    rows = []
    for meth, s in scores(tr, te).items():
        t = te.assign(s=s)
        fv = t.groupby(["bat", "cycle"]).agg(s=("s", "mean"), capacite=("capacite", "first")).reset_index()
        for b, g in fv.groupby("bat"):
            rows.append(dict(cadre=cadre, methode=meth, bat=b,
                             sp_capacite=spearmanr(g.s, -g.capacite).statistic,
                             sp_cycle=spearmanr(g.s, g.cycle).statistic, n_cycles=len(g)))
    return rows


def main():
    df = charger()
    groupe = {b: g for g, bs in GROUPES.items() for b in bs}
    df["groupe"] = df.bat.map(groupe)
    tr = df[df.groupe == "uniforme"]
    rows = []
    rows += mesurer(tr, df[df.groupe != "uniforme"], "classe non vue")
    for b in GROUPES["uniforme"]:
        rows += mesurer(tr[tr.bat != b], tr[tr.bat == b], "contrôle (uniforme laissée de côté)")
    t = pd.DataFrame(rows)
    t["groupe"] = t.bat.map(groupe)
    t.to_csv(v.OUT / "vB22_batteries.csv", index=False)
    s_tr = sfa.ajuster(tr[S_B].values, tr.bat.values, tr.cycle.values).score(tr[S_B].values)
    diag = dict(R_chapeau=r14.r_chapeau(s_tr, tr.bat.values, tr.cycle.values),
                D=d15.detecteur(s_tr, tr[W_B].values, tr.bat.values, tr.cycle.values))
    pd.set_option("display.width", 220)
    print("=== étapes par groupe ; cycles par batterie ===")
    print(df.groupby("groupe").agg(etapes=("bat", "size"), batteries=("bat", "nunique"),
                                   I_max=("I", "max")).to_string())
    ordre = ["sfa_brut", "sfa_resid_lin", "sfa_resid_quad", "span_W", "sonde"]
    for m in ["sp_capacite", "sp_cycle"]:
        print(f"\n=== {m} — médiane par groupe ===")
        print(t.groupby(["cadre", "groupe", "methode"])[m].median().unstack("methode")[ordre].round(2).to_string())
    print(f"\ndiagnostics sans étiquettes (SFA brute, batteries uniformes) : R̂ = {diag['R_chapeau']:.1f}, D = {diag['D']:.2f}")


if __name__ == "__main__":
    main()
