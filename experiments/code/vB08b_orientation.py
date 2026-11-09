"""vB08b — Orienter le score sans se laisser tromper par la dérive d'usage.

vB08 : sous dérive forte, sur DS07, `insens_W_v99` et `span_W` trouvent une direction juste mais
ORIENTÉE À L'ENVERS sur les trois unités, référence comprise (−0,74 à −0,94). L'orientation actuelle
(`sfa.orienter`) prend le signe de la corrélation entre le score et l'ordre des vols d'entraînement ;
sous dérive, la tendance du score à l'entraînement peut suivre la dérive plutôt que la dégradation.

Orientation « partielle », déclarée avant mesure : sur les vols d'entraînement, régression linéaire
    score_vol ~ constante par unité + rang du vol dans la vie + φ(W_vol)
(φ = W, W², produits, comme `span_W`) ; signe = signe du coefficient du rang. C'est la tendance du score
à conditions égales. Aucune étiquette. Seule l'orientation change ; la direction est celle de vB08.

Méthodes : `span_W` et `insens_W_v99` (vB08), chacune avec l'orientation d'origine et la partielle ;
`sans_W_v99` rapportée pour repère (elle n'a pas W, elle garde son orientation). Scénarios de vB05.

Lecture fixée AVANT de mesurer (Spearman par vol, référence + non vues) :
  - l'orientation partielle corrige l'inversion de DS07 (les trois unités ≥ 0,7) sous dérive forte ;
  - et ne crée aucune inversion nouvelle ailleurs (sans dérive et dérive modérée : mêmes valeurs).
  Elle ne peut rien pour DS01 (direction perdue) ni DS05 (direction mêlée) : on le vérifie.

Machine : Mac. Sortie : voie_B_avec_exo/results/vB08b.csv, vB08b.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402  (ajoute aussi ../code au chemin)
import vB05_derive_usage as d5  # noqa: E402
import vB05c_derive_regularisee as d5c  # noqa: E402
import vB08_espace_conditions as v8  # noqa: E402
import sfa  # noqa: E402


def signe_partiel(m: sfa.SFA, tr: pd.DataFrame) -> int:
    """+1 si le score croît avec l'âge à conditions égales (unités d'entraînement), −1 sinon."""
    s = m.score(tr[v.S_COLS].values)
    W = tr[v.W_COLS].values
    Wz = (W - W.mean(0)) / W.std(0)
    fv = sfa.moyennes_par_vol(np.column_stack([s, v8.phi(Wz)]), tr.unit.values, tr.cycle.values)
    rang = sfa.rang_dans_la_vie(fv)
    U = pd.get_dummies(fv.unit).values.astype(float)                    # constante par unité
    F = fv[list(range(1, fv.shape[1] - 2))].values
    F = (F - F.mean(0)) / (F.std(0) + 1e-12)
    X = np.column_stack([U, rang, F])
    coef, *_ = np.linalg.lstsq(X, fv[0].values, rcond=None)
    return 1 if coef[U.shape[1]] >= 0 else -1


def main():
    rows = []
    for ds in d5.DS_:
        v.it10.DILATION = v.parse_key(ds)[1]
        fc, train, ref, unseen = v.PLAN[ds]
        df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
        complets = {u: v.fenetres(df, u) for u in [ref] + list(unseen)}
        for nom_sc, derive in d5.SCENARIOS.items():
            tr = pd.concat([d5.fenetres_derive(df, u, derive)[0] for u in train], ignore_index=True)
            X, W, un, cy = tr[v.S_COLS].values, tr[v.W_COLS].values, tr.unit.values, tr.cycle.values
            base = {"span_W": v8.span_W(X, W, un, cy)[0], "insens_W_v99": d5c.insens_v99(X, W, un, cy),
                    "sans_W_v99": sfa.ajuster_reduite(X, un, cy)[0]}
            for nom, m in base.items():
                variantes = [("ordre", m)]
                if nom != "sans_W_v99":
                    sg = signe_partiel(m, tr)
                    variantes.append(("partielle", sfa.SFA(m.moyenne, m.ecart_type, sg * m.w)))
                for orient, mm in variantes:
                    res = v8.evaluer_modele(mm, complets, tr, ds, train, ref, unseen)
                    for _, r in res.iterrows():
                        rows.append(dict(DS=ds, scenario=nom_sc, methode=nom, orientation=orient,
                                         unite=r.unite, role=r.role, spearman=r.corr_spearman))
    t = pd.DataFrame(rows)
    t.to_csv(v.OUT / "vB08b.csv", index=False)
    pd.set_option("display.width", 250)
    p = t.pivot_table(index=["methode", "orientation", "scenario"], columns=["DS", "unite"],
                      values="spearman").round(2)
    print("=== Spearman par vol (référence incluse : DS01 u9, DS05 u10, DS07 u10) ===")
    print(p.to_string())
    ch = (t.pivot_table(index=["methode", "scenario", "DS", "unite"], columns="orientation",
                        values="spearman").dropna())
    ch = ch[np.sign(ch.ordre) != np.sign(ch.partielle)]
    print("\n=== cas où l'orientation partielle change le signe ===")
    print(ch.round(2).to_string() if len(ch) else "aucun")


if __name__ == "__main__":
    main()
