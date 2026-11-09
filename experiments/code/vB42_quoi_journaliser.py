"""vB42 — Quoi journaliser ? Attributs, granularité, fréquence, référence saine (R14 de PAPIER_ICDE § 8).

Relecture (W1) : la question « faut-il journaliser le contexte ? » doit être MESURÉE, pas seulement inférée. Quatre
volets, mêmes cas, données et mesure que vB39 (tâches de vB39, avec une liste d'attributs ou un chargement modifié) :
  (a) ABLATION des attributs de W : chaque sous-ensemble non vide des attributs de W (N-CMAPSS : alt, Mach, TRA, T2 ;
      C-MAPSS : 3 réglages, régimes retenus ; batteries : courant, charge tirée ; roulements : charges dynamique et
      statique, vitesse, température ambiante ; PVDAQ : irradiance, température ambiante ; gaz : un seul attribut,
      exclu) ; méthodes avec W continu et linéaires : `resid_lin`, `resid_quad`, `span_W` (définitions de vB35).
  (b) GRANULARITÉ des labels de régime : k-moyennes sur W standardisé avec K ∈ {2, 4, 6, 12, 24} (C-MAPSS aussi,
      en plus des régimes natifs) ; méthodes `norm_regime`, `invariant_regime` (vB33).
  (c) FRÉQUENCE : W journalisé une fois par vol (moyenne du vol, en entraînement comme en test) au lieu de chaque
      fenêtre ; jeux où un vol contient plusieurs fenêtres : N-CMAPSS, batteries, PVDAQ ; méthodes de (a) avec tout W.
  (d) C1 avec RÉFÉRENCE SAINE couvrant les contextes : scénarios de dérive d'usage de vB37 (N-CMAPSS, 12 découpages) ;
      en plus des données dérivées, on suppose journalisée une période de référence saine : tous les vols des 20 %
      premiers de la vie de chaque unité d'entraînement (sans sélection par la dérive). `resid_quad_ref` : modèle
      quadratique W → S appris sur la référence, SFA sur le résidu des données dérivées ; `span_W_ref` : sous-espace des
      effets du contexte estimé sur la référence, SFA dans son complément sur les données dérivées (statistiques de
      standardisation de la référence). Comparés à `lenteur`, `resid_quad`, `span_W` sur les données dérivées seules.

Lecture fixée AVANT de mesurer (par cas : écart apparié à la meilleure sans contexte de vB39 = max de lenteur,
tendance_moy, tendance_reg ; par jeu : médiane ; jeux informatifs ; PVDAQ rapporté sans lecture) :
  (a) un attribut « vaut d'être journalisé » sur un jeu si le meilleur sous-ensemble qui le contient, avec la meilleure
      des trois méthodes, dépasse la meilleure sans contexte de 0,05 ou plus. Rapporté aussi : valeur marginale de
      chaque attribut (médiane, sur les sous-ensembles qui ne le contiennent pas, du gain à l'ajouter). Attendu
      (déclaré) : seulement sur les batteries.
  (b) « la granularité change la conclusion » si une valeur de K donne une méthode à labels ≥ meilleure sans
      contexte + 0,05 sur un jeu informatif. Attendu : non.
  (c) « un W par vol suffit » sur un jeu si la meilleure des trois méthodes avec W par vol est à 0,05 au plus de la
      meilleure avec W par fenêtre (médianes). Pas d'attendu déclaré.
  (d) « une référence saine rend le contexte utile sous C1 » si max(resid_quad_ref, span_W_ref) ≥ lenteur + 0,05
      (médiane sur les 12 découpages, par scénario). Pas d'attendu déclaré.

Machine : choisie par Martin. Sortie : results/vB42_{ablation,granularite,frequence,c1}.csv, vB42.txt
Lancement : `python vB42_quoi_journaliser.py [ablation|granularite|frequence|c1|tout]`.
"""
from __future__ import annotations

import itertools
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402
import vB19_cmapss_avec_w as b19  # noqa: E402
import vB22_batteries as b22  # noqa: E402
import vB29_roulements_upb as u29  # noqa: E402
import vB31_pvdaq as p31  # noqa: E402
import vB39_banc_par_unite as b39  # noqa: E402

KS = [2, 4, 6, 12, 24]
# listes d'attributs de W de chaque jeu (module, nom de la constante)
LISTES = {"N": (v, "W_COLS"), "C": (b19, "W_C"), "B": (b22, "W_B"), "R": (u29, "W_U"), "P": (p31, "W_P")}
ORIG = {k: list(getattr(m, a)) for k, (m, a) in LISTES.items()}
ORIG_FEN, ORIG_B22, ORIG_PV = v.fenetres, b22.charger, p31.ch.charger


def restaurer():
    for k, (m, a) in LISTES.items():
        setattr(m, a, list(ORIG[k]))
    v.fenetres, b22.charger, p31.ch.charger = ORIG_FEN, ORIG_B22, ORIG_PV


# --------------------------------------------------------------------------------------------------
def lineaires(S, W, unit, cycle, labels=None):
    """resid_lin, resid_quad, span_W (définitions de vB35, colonnes de résidu sans variance retirées)."""
    import sfa
    import vB24_regle_choix as r24
    import vB27_robustesse_decoupages as r27
    if W.shape[1] == 0:
        return {}
    out = {}
    m = r27.v8.span_W(S, W, unit, cycle, reduire=False)[0]
    out["span_W"] = lambda S_, W_, m=m: m.score(S_)
    muW, sdW = W.mean(0), W.std(0) + 1e-12
    for nom, base in [("resid_lin", lambda W_: (W_ - muW) / sdW), ("resid_quad", lambda W_: r24.phi((W_ - muW) / sdW))]:
        reg = LinearRegression().fit(base(W), S)
        E = S - reg.predict(base(W))
        sdE = E.std(0)
        garde = sdE > 1e-6 * sdE.max()
        mm = sfa.ajuster(E[:, garde], unit, cycle)
        out[nom] = (lambda S_, W_, m=mm, reg=reg, base=base, garde=garde:
                    m.score((S_ - reg.predict(base(W_)))[:, garde]))
    return out


def granulaires(S, W, unit, cycle, labels=None):
    import vB33_labels_regime as m33
    out = {}
    for K in KS:
        m33.K = K
        g = m33.methodes(S, W, unit, cycle, labels=None)
        out[f"norm_regime_K{K}"], out[f"invariant_regime_K{K}"] = g["norm_regime"], g["invariant_regime"]
    m33.K = 6
    if labels is not None:                                 # C-MAPSS : régimes natifs
        g = m33.methodes(S, W, unit, cycle, labels=labels)
        out["norm_regime_natif"], out["invariant_regime_natif"] = g["norm_regime"], g["invariant_regime"]
    return out


# --------------------------------------------------------------------------------------------------
def _par_vol(df, cles, cols):
    df = df.copy()
    df[cols] = df.groupby(cles)[cols].transform("mean")
    return df


def fen_par_vol(*a, **k):
    f = ORIG_FEN(*a, **k)
    return _par_vol(f, ["unit", "cycle"], ORIG["N"])


def b22_par_vol():
    return _par_vol(ORIG_B22(), ["bat", "cycle"], ORIG["B"])


def pv_par_vol():
    return _par_vol(ORIG_PV(), ["unit", "cycle"], ORIG["P"])


def w_ablation(t):
    tache, sous = t
    restaurer()
    m, a = LISTES[tache[0]]
    setattr(m, a, list(sous))
    b39.tout = lineaires
    rows = b39.lancer(tache)
    restaurer()
    return [dict(r, attributs="+".join(sous)) for r in rows]


def w_granularite(tache):
    restaurer()
    b39.tout = granulaires
    return b39.lancer(tache)


def w_frequence(t):
    tache, mode = t
    restaurer()
    if mode == "vol":
        v.fenetres, b22.charger, p31.ch.charger = fen_par_vol, b22_par_vol, pv_par_vol
    b39.tout = lineaires
    rows = b39.lancer(tache)
    restaurer()
    return [dict(r, frequence=mode) for r in rows]


def w_c1(args):
    """Comme vB37.tache, avec une référence saine (20 % premiers de la vie, tous les vols) en plus."""
    import sfa
    import vB05_derive_usage as d5
    import vB08_espace_conditions as v8
    import vB24_regle_choix as r24
    import vB27_robustesse_decoupages as r27
    from evaluateur import evaluer
    ds, ref, sc = args
    v.it10.DILATION = v.parse_key(ds)[1]
    df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
    fc = df.groupby("unit").Fc.first()
    train = [u for u in sorted(fc[fc == 1].index) if u != ref]
    unseen = sorted(fc[fc != 1].index)
    complets = {u: v.fenetres(df, u) for u in [ref] + unseen}
    tr = pd.concat([d5.fenetres_derive(df, u, d5.SCENARIOS[sc])[0] for u in train], ignore_index=True)
    rf = []
    for u in train:
        du = df[df.unit == u]
        cy = np.sort(du.cycle.unique())
        rf.append(v.fenetres(du[du.cycle.isin(cy[: max(2, int(0.2 * len(cy)))])], u))
    rf = pd.concat(rf, ignore_index=True)
    S, W, un, cyc = tr[v.S_COLS].values, tr[v.W_COLS].values, tr.unit.values, tr.cycle.values
    Sr, Wr = rf[v.S_COLS].values, rf[v.W_COLS].values
    fs = r27.methodes(S, W, un, cyc)
    fs["resid_quad"] = fs.pop("modele")
    # résidu quadratique appris sur la référence
    muW, sdW = Wr.mean(0), Wr.std(0) + 1e-12
    q = LinearRegression().fit(r24.phi((Wr - muW) / sdW), Sr)
    E = S - q.predict(r24.phi((W - muW) / sdW))
    sdE = E.std(0)
    garde = sdE > 1e-6 * sdE.max()
    m_q = sfa.ajuster(E[:, garde], un, cyc)
    fs["resid_quad_ref"] = lambda S_, W_: m_q.score((S_ - q.predict(r24.phi((W_ - muW) / sdW)))[:, garde])
    # span_W : sous-espace estimé sur la référence, SFA dans le complément sur les données dérivées
    _, info = v8.span_W(Sr, Wr, rf.unit.values, rf.cycle.values, reduire=False)
    k, V = S.shape[1], info["V"]
    N = np.linalg.svd(np.eye(k) - V @ V.T)[0][:, :k - V.shape[1]]
    mu_r, sd_r = Sr.mean(0), Sr.std(0)
    fv = sfa.moyennes_par_vol((S - mu_r) / sd_r, un, cyc)
    w = v8.sfa_complement(fv, k, N)
    m_s = sfa.SFA(moyenne=mu_r, ecart_type=sd_r, w=w)
    fs["span_W_ref"] = lambda S_, W_: m_s.score(S_)
    row = dict(DS=ds, ref=ref, scenario=sc)
    for nom, f in fs.items():
        fen = pd.concat([x.assign(score=f(x[v.S_COLS].values, x[v.W_COLS].values))
                         for x in list(complets.values()) + [tr]], ignore_index=True)
        res = evaluer(fen, ds=ds, train_units=train, ref_unit=ref, non_vues=unseen)
        row[nom] = float(res[res.role == "non_vue"].corr_spearman.median())
    return [row]


# --------------------------------------------------------------------------------------------------
def taches_base(sans=()):
    import vB05_derive_usage as d5
    t = []
    for ds in d5.DS_:
        fc = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv", usecols=["unit", "Fc"]).groupby("unit").Fc.first()
        t += [("N", (ds, ref)) for ref in sorted(fc[fc == 1].index)]
    t += [("C", (fd, cas, ret, cib, k)) for fd in ["FD002", "FD004"] for cas, ret, cib in b19.a6.CAS
          for k in range(b19.N_TIRAGES) if ret is not None]
    t += [("B", g) for g in b22.GROUPES] + [("R", g) for g in u29.GROUPES]
    t += [("G", (gz, imp)) for gz in range(1, 7) for imp in (True, False)]
    t += [("P", (a, b, u)) for a, b in [("Las_Vegas", "Golden"), ("Golden", "Las_Vegas")] for u in p31.SITES[a]]
    return [x for x in t if x[0] not in sans]


def sous_ensembles(cols):
    return [c for r in range(1, len(cols) + 1) for c in itertools.combinations(cols, r)]


def par_cas(rows, cles):
    L = pd.DataFrame(rows)
    C = L.groupby(["jeu", "cas", "methode"] + cles).rho.median().reset_index()
    C.loc[C.jeu == "C-MAPSS", "jeu"] = "C-MAPSS retenu"
    return C


def sans_contexte():
    r = pd.read_csv(v.OUT / "vB39_cas.csv")
    r.loc[(r.jeu == "C-MAPSS") & r.cas.str.contains(r" R\d"), "jeu"] = "C-MAPSS retenu"
    r["cf"] = r[["lenteur", "tendance_moy", "tendance_reg"]].max(axis=1)
    return r[["jeu", "cas", "cf"]]


def main():
    parties = sys.argv[1:] or ["tout"]
    if "tout" in parties:
        parties = ["ablation", "granularite", "frequence", "c1"]
    n_proc = int(os.environ.get("VB42_PROC", "10"))
    seul = os.environ.get("VB42_SEUL")                     # essai rapide : une tâche de ce jeu
    pd.set_option("display.width", 250)
    fh = open(v.OUT / ("vB42.txt" if not seul else f"vB42_essai_{seul}.txt"), "a")

    def p(*a):
        print(*a)
        print(*a, file=fh)
    cf = sans_contexte() if (v.OUT / "vB39_cas.csv").exists() else None
    inf = ["N-CMAPSS", "C-MAPSS retenu", "batteries", "roulements", "gaz"]

    def lourds_dabord(t):
        return sorted(t, key=lambda x: 0 if x[0][0] in ("N", "R") else 1)

    if "ablation" in parties:
        T = [(t, s) for t in taches_base(sans=("G",)) for s in sous_ensembles(ORIG[t[0]])]
        if seul:
            T = [x for x in T if x[0][0] == seul][:3]
        with Pool(n_proc) as pool:
            rows = [r for rr in pool.map(w_ablation, lourds_dabord(T), chunksize=1) for r in rr]
        C = par_cas(rows, ["attributs"])
        C.to_csv(v.OUT / f"vB42_ablation{'_essai' if seul else ''}.csv", index=False)
        p("\n=== (a) ablation : médiane par jeu, meilleure des 3 méthodes par sous-ensemble ===")
        M = C.groupby(["jeu", "attributs", "methode"]).rho.median().unstack("methode")
        M["meilleure"] = M.max(axis=1)
        p(M.round(2).to_string())
        if cf is not None:
            G = C.pivot_table(index=["jeu", "cas", "attributs"], columns="methode", values="rho").max(axis=1)
            G = G.reset_index(name="avec").merge(cf, on=["jeu", "cas"])
            G["gain"] = G.avec - G.cf
            gm = G.groupby(["jeu", "attributs"]).gain.median()
            p("\nmédiane des gains appariés sur la meilleure sans contexte, par sous-ensemble :")
            p(gm.round(3).to_string())
            for j in gm.index.get_level_values(0).unique():
                for a in ORIG[{"N-CMAPSS": "N", "C-MAPSS retenu": "C", "batteries": "B", "roulements": "R",
                               "PVDAQ": "P"}[j]]:
                    best = max(g for s, g in gm.loc[j].items() if a in s.split("+"))
                    marg = [gm.loc[j].get("+".join(sorted(set(s.split("+")) | {a}, key=ORIG_ORDER(j).index)), np.nan)
                            - g for s, g in gm.loc[j].items() if a not in s.split("+")]
                    p(f"  {j:16s} {a:8s} meilleur gain avec lui {best:+.3f} → "
                      f"{'vaut d’être journalisé' if best >= 0.05 and j in inf else 'non'} ; "
                      f"valeur marginale médiane {np.nanmedian(marg) if marg else np.nan:+.3f}")
    if "granularite" in parties:
        T = taches_base(sans=())
        if seul:
            T = [x for x in T if x[0] == seul][:1]
        with Pool(n_proc) as pool:
            rows = [r for rr in pool.map(w_granularite, lourds_dabord(T), chunksize=1) for r in rr]
        C = par_cas(rows, [])
        C.to_csv(v.OUT / f"vB42_granularite{'_essai' if seul else ''}.csv", index=False)
        P = C.pivot_table(index=["jeu", "cas"], columns="methode", values="rho").reset_index()
        p("\n=== (b) granularité : médiane par jeu ===")
        p(P.groupby("jeu").median(numeric_only=True).T.round(2).to_string())
        if cf is not None:
            P = P.merge(cf, on=["jeu", "cas"])
            for K in KS + ["natif"]:
                cols = [c for c in (f"norm_regime_K{K}", f"invariant_regime_K{K}",
                                    f"norm_regime_{K}", f"invariant_regime_{K}") if c in P]
                if not cols:
                    continue
                g = (P[cols].max(axis=1) - P.cf).groupby(P.jeu).median()
                p(f"  K = {K!s:6s} gain médian des labels : " + " ; ".join(f"{j} {x:+.3f}" for j, x in g.items()))
    if "frequence" in parties:
        T = [(t, m) for t in taches_base() if t[0] in ("N", "B", "P") for m in ("fenetre", "vol")]
        if seul:
            T = [x for x in T if x[0][0] == seul][:2]
        with Pool(n_proc) as pool:
            rows = [r for rr in pool.map(w_frequence, lourds_dabord(T), chunksize=1) for r in rr]
        C = par_cas(rows, ["frequence"])
        C.to_csv(v.OUT / f"vB42_frequence{'_essai' if seul else ''}.csv", index=False)
        M = C.groupby(["jeu", "frequence", "methode"]).rho.median().unstack("methode")
        M["meilleure"] = M.max(axis=1)
        p("\n=== (c) fréquence de W : médiane par jeu ===")
        p(M.round(2).to_string())
        for j in M.index.get_level_values(0).unique():
            e = M.loc[(j, "vol"), "meilleure"] - M.loc[(j, "fenetre"), "meilleure"]
            p(f"  {j:12s} W par vol − W par fenêtre : {e:+.3f} → {'un W par vol suffit' if e >= -0.05 else 'non'}")
    if "c1" in parties:
        import vB05_derive_usage as d5
        import vB37_derive_span_w as b37
        T = []
        for ds in d5.DS_:
            fc = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv", usecols=["unit", "Fc"]).groupby("unit").Fc.first()
            T += [(ds, ref, sc) for ref in sorted(fc[fc == 1].index) for sc in b37.DERIVES]
        if seul:
            T = T[:1]
        with Pool(n_proc) as pool:
            t = pd.DataFrame([r for rr in pool.map(w_c1, T, chunksize=1) for r in rr])
        t.to_csv(v.OUT / f"vB42_c1{'_essai' if seul else ''}.csv", index=False)
        meth = ["lenteur", "resid_quad", "span_W", "resid_quad_ref", "span_W_ref"]
        med = t.groupby("scenario")[meth].median()
        p("\n=== (d) C1 avec référence saine : médiane sur les découpages ===")
        p(med.round(2).to_string())
        for s, r in med.iterrows():
            ok = max(r.resid_quad_ref, r.span_W_ref) >= r.lenteur + 0.05
            p(f"  {s:20s} → {'une référence saine rend le contexte utile' if ok else 'non'}")
    fh.close()


def ORIG_ORDER(jeu):
    return ORIG[{"N-CMAPSS": "N", "C-MAPSS retenu": "C", "batteries": "B", "roulements": "R", "PVDAQ": "P"}[jeu]]


if __name__ == "__main__":
    main()
