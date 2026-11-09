"""vB51 — E1 : changements de collecte injectés, et valeur d'un journal d'événements (PAPIER_ICDE § 9, CANAL M-34).

Question : un changement de collecte (recalibration, changement d'unité, canaux permutés, capteur remplacé) fait-il
échouer le suivi de dégradation sans étiquettes, et un journal qui en donne les dates (sans dire quel capteur) permet-il
de le corriger ? Avec quelle complétude et quelle précision de date ?

Jeux et cas (vérité terrain par cycle, découpages hors support de vB39) :
  - N-CMAPSS DS01/05/07, 12 cas (vérité : dégradation simulée, `evaluateur.etiquettes`) ;
  - C-MAPSS FD002/FD004, 30 cas à régime retenu (vérité : cycle, évaluée sur le régime cible) ;
  - batteries, 5 cas (entraînement sur un groupe, test sur toutes les cellules des autres groupes ; vérité : −capacité) ;
  - roulements, 4 cas (entraînement sur un groupe, test sur les autres hors B01, B05 ; vérité : cycle).
Mesure : score moyen par cycle, Spearman avec la vérité par unité de test, médiane des unités = valeur du cas.
(Évaluation propre à E1, la même pour toutes les conditions ; seules les comparaisons entre conditions comptent.)

Événements injectés (entraînement ET test, tirages indépendants par unité) : 2 par unité, à des cycles tirés
uniformément entre 10 et 90 % de sa vie, chacun sur un capteur tiré au hasard ; effet sur toutes les lignes à partir
du cycle de l'événement. Quatre types, un à la fois :
  - `decalage` (recalibration) : ± 3 σ_bruit du capteur (signe tiré au hasard) (σ_bruit = médiane, sur les unités d'entraînement, de l'écart-type
    des différences entre cycles consécutifs des moyennes par cycle, calculée sur les données propres) ;
  - `unite` (changement d'unité) : valeur brute × 10 ou × 0,1 (au hasard) ;
  - `permutation` : le capteur échange ses valeurs avec un autre capteur tiré au hasard ;
  - `bruit` (capteur remplacé, plus bruité) : + bruit gaussien d'écart-type 3 σ_bruit sur chaque fenêtre.
Journal : seulement des DATES (pas le capteur). Conditions :
  propre (aucun événement) ; sans journal ; journal complet exact ; complétude 50 % ; dates fausses de ±5 cycles ;
  ±20 cycles ; complet exact + 1 fausse entrée par unité.
Corrections avec journal (méthodes génériques), fenêtre k = 5 % des cycles de l'unité (au moins 10) :
  - `realignement` : à chaque date du journal, pour chaque capteur, médiane et écart absolu médian (×1,4826) sur les k
    cycles avant et après ; si le rapport des échelles dépasse 3 → les lignes postérieures sont ramenées sur l'échelle et
    le niveau d'avant (changement d'unité) ; sinon, si le saut de niveau dépasse 4 erreurs-types → il est retiré ; sinon
    rien. Le journal donne la date, les données disent quel capteur a changé.
    VERSION 2 (15 oct.) : la version 1 (réalignement affine de TOUS les capteurs sur 10 cycles) a été remplacée après un
    essai sur un cas batteries et un cas roulements, avant tout lancement complet : elle injectait du bruit sur les
    capteurs intacts (span_W des roulements 0,82 → −0,08 « avec » journal). Défaut de la correction, pas une lecture.
    Même essai : les sauts injectés allaient toujours dans le même sens (+3σ, ×10) et tard dans la vie, donc une marche
    croissante avec l'âge, corrélée à toute vérité monotone (l'événement « aidait » parfois). Signe tiré au hasard.
    CORRECTION DU 19 OCT. (bug) : l'écart absolu médian vaut 0 pour les capteurs de C-MAPSS qui ne prennent que
    quelques valeurs ; le test d'échelle divisait alors par ~0 et « remettait à l'échelle » des capteurs intacts
    (jusqu'à 10^14 σ, échecs LinAlgError en aval). Dispersion = écart-type si l'écart absolu médian est nul, test
    d'échelle seulement si les deux dispersions sont non nulles. Les résultats C-MAPSS antérieurs au 19 oct. sont faux.
  - `exclusion` : les lignes à moins de k cycles d'une date du journal sont retirées (entraînement et évaluation) ;
    seulement pour le journal complet exact.
Méthodes : lenteur, tendance_reg (vB32), resid_quad, span_W (vB42.lineaires). 2 tirages d'événements par cas.

Lecture (usage interne, fixée avant de mesurer), par jeu et méthode, médiane sur cas et tirages :
  (1) un type d'événement « fait échouer » la méthode si (sans journal − propre) ≤ −0,05 ;
  (2) le journal « corrige » si (journal complet exact, réalignement − sans journal) ≥ +0,05 ; part récupérée =
      (avec journal − sans) / (propre − sans) ;
  (3) qualité : part récupérée à complétude 50 %, dates ±5, ±20, fausse entrée.
Attendus (déclarés) : décalage et unité font échouer et sont corrigés par réalignement ; permutation non corrigée par le
réalignement (il ne rend pas l'identité des canaux) ; bruit peu nuisible à la lenteur sur des moyennes par cycle.

Machine : Mac. Sortie : results/vB51_cas.csv, vB51.txt
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vB01_sonde_marge as v  # noqa: E402

TYPES = ["decalage", "unite", "permutation", "bruit"]
CONDITIONS = ["sans_journal", "journal", "complet50", "date5", "date20", "fausse"]
K, N_EV, TIRAGES = 10, 2, (0, 1)


# --------------------------------------------------------------------------------------------------
# Extraction générique des cas : (entraînement, test, colonnes S, colonnes W) ; colonnes unit, cycle, verite, eval
def cas_ncmapss(ds, ref):
    from evaluateur import etiquettes
    import vB05_derive_usage as d5
    v.it10.DILATION = v.parse_key(ds)[1]
    df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
    fc = df.groupby("unit").Fc.first()
    train = [u for u in sorted(fc[fc == 1].index) if u != ref]
    unseen = sorted(fc[fc != 1].index)
    lab = etiquettes(ds)[["unit", "cycle", "degradation"]].rename(columns={"degradation": "verite"})
    tr = pd.concat([d5.fenetres_derive(df, u, None)[0] for u in train], ignore_index=True)
    te = pd.concat([v.fenetres(df, u) for u in unseen], ignore_index=True)
    tr, te = tr.merge(lab, on=["unit", "cycle"]), te.merge(lab, on=["unit", "cycle"])
    return tr.assign(eval=True), te.assign(eval=True), list(v.S_COLS), list(v.W_COLS)


def cas_cmapss(fd, retire, tirage):
    import vB19_cmapss_avec_w as b19
    d = b19.a6.charger(fd)
    rng = np.random.default_rng(tirage)
    u = d.unit.unique()
    tr_u = set(rng.choice(u, size=len(u) // 2, replace=False))
    tr = d[d.unit.isin(tr_u) & (d.regime != retire)]
    te = d[~d.unit.isin(tr_u)]
    X = [c for c in b19.a6.COLS[5:] if tr[c].std() > 1e-8]
    Wk = [c for c in b19.W_C if tr[c].std() > 1e-8]
    tr = tr.assign(verite=tr.cycle, eval=True)
    te = te.assign(verite=te.cycle, eval=(te.regime == retire).values)
    return tr, te, X, Wk


def cas_batteries(g_tr):
    import vB22_batteries as b22
    df = b22.charger()
    df["groupe"] = df.bat.map({b: g for g, bs in b22.GROUPES.items() for b in bs})
    df = df.rename(columns={"bat": "unit"}).assign(verite=lambda x: -x.capacite, eval=True)
    return df[df.groupe == g_tr], df[df.groupe != g_tr], list(b22.S_B), list(b22.W_B)


def cas_roulements(g_tr):
    import vB29_roulements_upb as u29
    df = u29.charger()
    tr = df[df.groupe == g_tr]
    te = df[(df.groupe != g_tr) & ~df.unit.isin(u29.ENTR_SEULEMENT)]
    Wk = [c for c in u29.W_U if c == "T_amb" or tr[f"set_{c}"].nunique() > 1]
    return tr.assign(verite=tr.cycle, eval=True), te.assign(verite=te.cycle, eval=True), list(u29.S_U), Wk


# --------------------------------------------------------------------------------------------------
def sigma_bruit(tr, S):
    fv = tr.groupby(["unit", "cycle"])[S].mean().reset_index().sort_values(["unit", "cycle"])
    return fv.groupby("unit")[S].apply(lambda g: g.diff().std()).median().values


def tirer_evenements(df, S, rng):
    ev = {}
    for u, g in df.groupby("unit"):
        cy = np.sort(g.cycle.unique())
        lo, hi = cy[int(0.1 * len(cy))], cy[max(int(0.9 * len(cy)) - 1, int(0.1 * len(cy)))]
        dates = np.sort(rng.choice(cy[(cy >= lo) & (cy <= hi)], N_EV, replace=False))
        ev[u] = [(int(t), int(rng.integers(len(S))), int(rng.integers(len(S))), int(rng.choice([-1, 1])))
                 for t in dates]
    return ev


def injecter(df, S, ev, type_, sig, rng):
    df = df.copy()
    X = df[S].values.astype(float)
    unit, cyc = df.unit.values, df.cycle.values
    for u, liste in ev.items():
        for t, j, j2, sg in liste:
            m = (unit == u) & (cyc >= t)
            if type_ == "decalage":
                X[m, j] += sg * 3 * sig[j]
            elif type_ == "unite":
                X[m, j] *= 10.0 ** sg
            elif type_ == "permutation":
                if j2 == j:
                    j2 = (j + 1) % len(S)
                X[m, j], X[m, j2] = X[m, j2].copy(), X[m, j].copy()
            elif type_ == "bruit":
                X[m, j] += rng.normal(0, 3 * sig[j], m.sum())
    df[S] = X
    return df


def journal(ev, cond, cycles, rng):
    """Dates journalisées par unité selon la condition (sans capteur)."""
    out = {}
    for u, liste in ev.items():
        dates = [t for t, *_ in liste]
        if cond == "complet50":
            dates = [t for t in dates if rng.random() < 0.5]
        if cond in ("date5", "date20"):
            e = 5 if cond == "date5" else 20
            dates = [t + int(rng.integers(-e, e + 1)) for t in dates]
        if cond == "fausse":
            cy = cycles[u]
            dates = dates + [int(rng.choice(cy[int(0.1 * len(cy)):int(0.9 * len(cy))]))]
        out[u] = sorted(dates)
    return out


def fenetre(cycles_u):
    return max(K, int(0.05 * len(cycles_u)))


def echelle(x):
    """Dispersion robuste : écart absolu médian × 1,4826, écart-type s'il est nul (capteurs à peu de valeurs)."""
    s = 1.4826 * np.median(np.abs(x - np.median(x)))
    return float(s if s > 0 else np.std(x))


def realigner(df, S, dates):
    """Correction à date connue, capteur par capteur, seulement si un saut y est détecté (version 2, voir en-tête)."""
    df = df.copy()
    X = df[S].values.astype(float)
    unit, cyc = df.unit.values, df.cycle.values
    for u, ds_ in dates.items():
        mu_u = unit == u
        k = fenetre(np.unique(cyc[mu_u]))
        for t in ds_:
            avant = mu_u & (cyc >= t - k) & (cyc < t)
            apres = mu_u & (cyc >= t) & (cyc < t + k)
            if avant.sum() < 3 or apres.sum() < 3:
                continue
            post = mu_u & (cyc >= t)
            for j in range(X.shape[1]):
                a, p = X[avant, j], X[apres, j]
                ma, mp = np.median(a), np.median(p)
                sa, sp = echelle(a), echelle(p)
                if sa > 0 and sp > 0 and abs(np.log(sp / sa)) > np.log(3):   # changement d'échelle (unité)
                    X[post, j] = (X[post, j] - mp) * (sa / sp) + ma
                elif abs(mp - ma) / (np.sqrt(sa ** 2 / avant.sum() + sp ** 2 / apres.sum()) + 1e-12) > 4:   # saut de niveau
                    X[post, j] += ma - mp
    df[S] = X
    return df


def exclure(df, dates):
    garde = np.ones(len(df), bool)
    for u, ds_ in dates.items():
        m = df.unit.values == u
        k = fenetre(np.unique(df.cycle.values[m]))
        for t in ds_:
            garde &= ~(m & (np.abs(df.cycle.values - t) < k))
    return df[garde]


ERREURS: list = []


def mesurer(tr, te, S, W):
    """Chaque méthode est ajustée et notée dans son propre bloc : un échec (données dégénérées) vaut NaN et est consigné."""
    import vB32_tendance_contre_lenteur as t32
    import vB42_quoi_journaliser as b42
    tev = te[te["eval"]]
    out = {k: np.nan for k in ["lenteur", "tendance_reg", "resid_quad", "span_W"]}

    def noter(k, construire):
        try:
            f = construire()
            sc = f(tev[S].values, tev[W].values)
            if not np.all(np.isfinite(sc)):
                raise ValueError("score non fini")
            s = tev.assign(s=sc).groupby(["unit", "cycle"]).agg(s=("s", "mean"), v=("verite", "first"))
            out[k] = float(np.median([spearmanr(g.s, g.v).statistic for _, g in s.groupby(level=0) if len(g) > 2]))
        except Exception as e:                              # noqa: BLE001 — consigné, jamais silencieux
            ERREURS.append(f"{k}: {type(e).__name__}: {str(e)[:80]}")
    t = {}

    def tendances():
        if not t:
            t.update(t32.methodes(tr[S].values, tr.unit.values, tr.cycle.values))
        return t
    lin = {}

    def lineaires():
        if not lin:
            lin.update(b42.lineaires(tr[S].values, tr[W].values, tr.unit.values, tr.cycle.values))
        return lin
    noter("lenteur", lambda: (lambda S_, W_: tendances()["lenteur"](S_)))
    noter("tendance_reg", lambda: (lambda S_, W_: tendances()["tendance_reg"](S_)))
    noter("resid_quad", lambda: lineaires()["resid_quad"])
    noter("span_W", lambda: lineaires()["span_W"])
    return out


def tache(args):
    jeu, cas, fonction, params = args
    tr, te, S, W = {"N": cas_ncmapss, "C": cas_cmapss, "B": cas_batteries, "R": cas_roulements}[fonction](*params)
    sig = sigma_bruit(tr, S)
    rows = [dict(jeu=jeu, cas=cas, type="propre", tirage=-1, condition="propre", correction="-", **mesurer(tr, te, S, W))]
    cycles = {u: np.sort(g.cycle.unique()) for u, g in pd.concat([tr, te]).groupby("unit")}
    for type_ in TYPES:
        for tg in TIRAGES:
            rng = np.random.default_rng(1000 * tg + TYPES.index(type_))
            ev_tr, ev_te = tirer_evenements(tr, S, rng), tirer_evenements(te, S, rng)
            tr_i, te_i = injecter(tr, S, ev_tr, type_, sig, rng), injecter(te, S, ev_te, type_, sig, rng)
            base = dict(jeu=jeu, cas=cas, type=type_, tirage=tg)
            rows.append(dict(base, condition="sans_journal", correction="-", **mesurer(tr_i, te_i, S, W)))
            for cond in CONDITIONS[1:]:
                rj = np.random.default_rng(7 + 1000 * tg + CONDITIONS.index(cond))
                j_tr, j_te = journal(ev_tr, cond, cycles, rj), journal(ev_te, cond, cycles, rj)
                rows.append(dict(base, condition=cond, correction="realignement",
                                 **mesurer(realigner(tr_i, S, j_tr), realigner(te_i, S, j_te), S, W)))
                if cond == "journal":
                    rows.append(dict(base, condition=cond, correction="exclusion",
                                     **mesurer(exclure(tr_i, j_tr), exclure(te_i, j_te), S, W)))
    if ERREURS:
        print(f"{jeu} {cas} : {len(ERREURS)} échecs, ex. {sorted(set(ERREURS))[:3]}", flush=True)
    return rows


def taches():
    import vB05_derive_usage as d5
    import vB19_cmapss_avec_w as b19
    import vB22_batteries as b22
    import vB29_roulements_upb as u29
    t = []
    for ds in d5.DS_:
        fc = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv", usecols=["unit", "Fc"]).groupby("unit").Fc.first()
        t += [("N-CMAPSS", f"{ds} ref u{r}", "N", (ds, r)) for r in sorted(fc[fc == 1].index)]
    t += [("C-MAPSS", f"{fd} {c} tirage {k}", "C", (fd, ret, k)) for fd in ["FD002", "FD004"]
          for c, ret, _ in b19.a6.CAS if ret is not None for k in range(b19.N_TIRAGES)]
    t += [("batteries", f"entr. {g}", "B", (g,)) for g in b22.GROUPES]
    t += [("roulements", f"entr. {g}", "R", (g,)) for g in u29.GROUPES]
    return t


def main():
    T = taches()
    seul = os.environ.get("VB51_SEUL")
    if seul:
        T = [x for x in T if x[2] == seul][:1]
    T = sorted(T, key=lambda x: 0 if x[2] in ("N", "R") else 1)
    with Pool(int(os.environ.get("VB51_PROC", "10"))) as pool:
        R = pd.DataFrame([r for rr in pool.map(tache, T, chunksize=1) for r in rr])
    suf = f"_essai_{seul}" if seul else ""
    R.to_csv(v.OUT / f"vB51_cas{suf}.csv", index=False)
    meths = ["lenteur", "tendance_reg", "resid_quad", "span_W"]
    L = R.melt(id_vars=["jeu", "cas", "type", "tirage", "condition", "correction"], value_vars=meths,
               var_name="methode", value_name="rho")
    propre = L[L.type == "propre"].set_index(["jeu", "cas", "methode"]).rho
    L = L[L.type != "propre"].copy()
    L["propre"] = propre.reindex(pd.MultiIndex.from_frame(L[["jeu", "cas", "methode"]])).values
    sans = L[L.condition == "sans_journal"].set_index(["jeu", "cas", "type", "tirage", "methode"]).rho
    L["sans"] = sans.reindex(pd.MultiIndex.from_frame(L[["jeu", "cas", "type", "tirage", "methode"]])).values
    lignes = []
    for (j, ty, m), g in L.groupby(["jeu", "type", "methode"]):
        r = dict(jeu=j, type=ty, methode=m, propre=g.propre.median(), sans=g[g.condition == "sans_journal"].rho.median())
        r["degat"] = float((g[g.condition == "sans_journal"].rho - g[g.condition == "sans_journal"].propre).median())
        for cond in CONDITIONS[1:]:
            x = g[(g.condition == cond) & (g.correction == "realignement")]
            r[f"gain_{cond}"] = float((x.rho - x.sans).median())
        x = g[(g.condition == "journal") & (g.correction == "exclusion")]
        r["gain_exclusion"] = float((x.rho - x.sans).median())
        lignes.append(r)
    T_ = pd.DataFrame(lignes)
    T_["part_recuperee"] = T_.gain_journal / (-T_.degat).where(T_.degat < -0.05)
    pd.set_option("display.width", 250)
    txt = T_.round(3).to_string(index=False)
    print(txt)
    (v.OUT / f"vB51{suf}.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
