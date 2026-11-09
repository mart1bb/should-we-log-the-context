"""vB06 — Option A : une SFA moins gourmande en vols.

Constat (vB05b) : la SFA commune (`../code/sfa.py`, 14 capteurs) s'effondre avec ~25 vols par unité
d'entraînement, même sans dérive (Spearman non vues 0,2–0,7), et tient avec ~90 (0,92–0,99).
Hypothèse : l'estimation de la direction la plus lente en 14 dimensions est bruitée ; le blanchiment
amplifie les directions de faible variance, où le bruit d'estimation domine, et la « plus lente » y
tombe par hasard.

Régularisation, déclarée avant mesure : réduire d'abord les capteurs standardisés (moyennes par vol
des unités d'entraînement) à leurs q premières composantes principales, puis SFA commune dans cet
espace, orientation par l'ordre des vols. Aucune étiquette ; capteurs seuls (comme la SFA de la voie A).
  - RÈGLE PRINCIPALE, fixée d'avance : q = plus petit nombre de composantes expliquant ≥ 99 % de la
    variance des moyennes par vol d'entraînement (`sfa_v99`) ;
  - SENSIBILITÉ (rapportée, jamais utilisée pour choisir) : q = 3, 5, 8, et 14 (= SFA commune).
Tailles d'entraînement : tous les vols ; ~45 et ~25 vols par unité, tirés au hasard sans dérive, mêmes
tailles que vB05 h = 0,30 et h = 0,15 (5 tirages chacune). Unités évaluées complètes.

Lecture fixée AVANT de mesurer (Spearman par vol, unités non vues, médiane des tirages) :
  - réussite : `sfa_v99` ≥ 0,85 à ~25 vols ET ≥ 0,9 avec tous les vols sur DS01, DS05, DS07 → la
    fragilité venait du bruit d'estimation ; on pourra refaire le test de dérive forte (vB05) avec elle,
    et proposer de l'ajouter au tronc commun ;
  - échec : ≈ SFA commune à ~25 vols → la réduction ne suffit pas.
Contrôle : q = 14 doit redonner la SFA commune (vB04 avec tous les vols).

Machine : Mac. Sortie : voie_B_avec_exo/results/vB06_*.csv, vB06.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402  (ajoute aussi ../code au chemin)
import vB05_derive_usage as d5  # noqa: E402
import sfa  # noqa: E402
from evaluateur import evaluer  # noqa: E402

N_TIRAGES = 5
VAR_CIBLE = sfa.VAR_CIBLE
Q_SENSIBILITE = [3, 5, 8, 14]


# promue au tronc commun le 24 sept. (CANAL M-05) : une seule implémentation, `../code/sfa.py`
ajuster_reduite = sfa.ajuster_reduite


def main():
    rng = np.random.default_rng(0)
    rows = []
    for ds in d5.DS_:
        v.it10.DILATION = v.parse_key(ds)[1]
        fc, train, ref, unseen = v.PLAN[ds]
        df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
        complets = {u: v.fenetres(df, u) for u in [ref] + list(unseen)}
        plans = {"tout": [None]}
        for nom, sc in [("~45", "montante_h0.30"), ("~25", "montante_h0.15")]:
            plans[nom] = [{u: len(d5.vols_gardes(df[df.unit == u], d5.SCENARIOS[sc])) for u in train}
                          for _ in range(N_TIRAGES)]
        for taille, tirages in plans.items():
            for k, tailles in enumerate(tirages):
                parts = []
                for u in train:
                    du = df[df.unit == u]
                    if tailles is not None:
                        vols = np.sort(rng.choice(du.cycle.unique(), tailles[u], replace=False))
                        du = du[du.cycle.isin(vols)]
                    parts.append(v.fenetres(du, u))
                tr = pd.concat(parts, ignore_index=True)
                args = (tr[v.S_COLS].values, tr.unit.values, tr.cycle.values)
                for q in [None] + Q_SENSIBILITE:
                    m, q_eff = ajuster_reduite(*args, q=q)
                    nom_m = "sfa_v99" if q is None else f"sfa_q{q}"
                    fen = pd.concat([x.assign(score=m.score(x[v.S_COLS].values))
                                     for x in list(complets.values()) + [tr]], ignore_index=True)
                    res = evaluer(fen, ds=ds, train_units=train, ref_unit=ref, non_vues=list(unseen))
                    for _, r in res.iterrows():
                        rows.append(dict(DS=ds, taille=taille, tirage=k, methode=nom_m, q=q_eff,
                                         unite=r.unite, role=r.role, spearman=r.corr_spearman,
                                         FA_classe=r.FA_classe, rappel=r.rappel))
    t = pd.DataFrame(rows)
    t.to_csv(v.OUT / "vB06_sfa_regularisee.csv", index=False)
    pd.set_option("display.width", 250)
    print("=== q retenu par la règle des 99 % (par DS et taille) ===")
    print(t[t.methode == "sfa_v99"].groupby(["DS", "taille"]).q.agg(["min", "max"]).to_string())
    ordre = ["sfa_v99"] + [f"sfa_q{q}" for q in Q_SENSIBILITE]
    for role in ["non_vue", "ref"]:
        print(f"\n=== Spearman par vol, {role} — médiane [min ; max] sur les tirages ===")
        g = t[t.role == role].groupby(["taille", "DS", "unite", "methode"]).spearman
        p = g.agg(lambda s: f"{s.median():.2f} [{s.min():.2f};{s.max():.2f}]"
                  if len(s) > 1 else f"{s.iloc[0]:.2f}").unstack("methode")[ordre]
        print(p.to_string())


if __name__ == "__main__":
    main()
