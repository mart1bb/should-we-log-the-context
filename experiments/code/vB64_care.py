"""vB64 — CARE to Compare (réel) : valeur du journal d'états à l'entraînement, à l'inférence et pour trier les alarmes ;
valeur du contexte rapide. Données : 95 jeux (3 parcs, 36 turbines), 44 avec une panne étiquetée, 51 normaux
(Gück et al. 2024, CC BY-SA 4.0), lus dans l'archive.

Journal = `status_type_id` à chaque pas de 10 min (0 normal, 1 bridage, 2 attente, 3 service, 4 arrêt, 5 autre) ;
« normal » = 0 ou 2 (README de CARE).
Capteurs suivis S = températures moyennes (description contenant « temperature »), sauf températures extérieures ou
ambiantes ; contexte W = vent, température extérieure, puissance active :
  A : wind_speed_3, sensor_0, power_30 ; B : wind_speed_61, sensor_8, power_62 ; C : wind_speed_236, sensor_7, power_6.
Lignes avec une valeur manquante dans S ou W écartées.
Scoreurs (ajustés sur l'entraînement du jeu) :
  - `ctx` : régression quadratique de chaque capteur sur W standardisé ; score d'une ligne = moyenne quadratique des
    résidus standardisés (écart-type des résidus d'entraînement) ;
  - `sans_ctx` : ACP sur S standardisé, 95 % de variance ; score = erreur de reconstruction quadratique moyenne.
Conditions (fixées avant de mesurer) :
  - entraînement : `tout` (toutes les lignes d'entraînement) ou `journal` (lignes à état normal) ;
  - inférence : `tout` ou `journal` (lignes à état anormal non notées) ;
  - score journalier = moyenne des lignes notées du jour (au moins 36) ; seuil = quantile 0,99 des scores journaliers
    d'entraînement, avec la même règle de notation ; jour en alarme si score > seuil.
Mesures par jeu : la période de prédiction de CARE se réduit à la fenêtre de l'événement (constaté sur l'essai du parc A,
avant le lancement complet : jeu 22, 8 jours, tous en arrêt ou en service). Fausses alarmes = part des jours en alarme
des jeux NORMAUX ; jeu anormal détecté si un jour noté de la fenêtre est en alarme (un jour sans ligne notée ne peut
pas l'être). Tri par le journal (entraînement `journal`, inférence `tout`) : un jour en alarme est EXPLIQUÉ si le jour même
ou la veille contient un état anormal ; part des fausses alarmes expliquées ; part des pannes dont tous les jours en
alarme de la fenêtre sont expliqués (masquées).
Contrôle sans seuil (ajouté le 19 oct. après la revue, avant de relancer) : par parc et condition, AUROC entre jeux
avec panne et jeux normaux du score journalier maximal de la fenêtre rapporté au seuil (jeux sans jour noté écartés,
comptés). Il dit si le gain de détection du journal à l'inférence vient d'une meilleure séparation ou seulement d'un
seuil plus bas.
Lecture (usage interne, fixée avant de mesurer), médianes sur jeux (fausses alarmes) et proportions (détection) :
  (1) journal à l'entraînement ou à l'inférence : « aide » si fausses alarmes −0,02 ou détection +5 points sans perte
      de l'autre ;
  (2) tri : « explique » si ≥ 50 % des fausses alarmes expliquées et ≤ 10 % des pannes masquées ;
  (3) contexte : ctx contre sans_ctx, mêmes critères.

Machine : Mac (données non fiables : lancer avec python -I). Sortie : results/vB64_jeux.csv, vB64.txt
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures  # noqa: E402

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))  # voir README
ARCHIVE = DATA_ROOT / "Wind_CARE_to_Compare/CARE_To_Compare.zip"
OUT = Path(__file__).resolve().parent.parent / "results"
W_PARC = {"A": ["wind_speed_3_avg", "sensor_0_avg", "power_30_avg"],
          "B": ["wind_speed_61_avg", "sensor_8_avg", "power_62_avg"],
          "C": ["wind_speed_236_avg", "sensor_7_avg", "power_6_avg"]}
NORMAL = {0, 2}
PAR_JOUR, MIN_JOUR = 144, 36


def meta(z, parc):
    racine = f"CARE_To_Compare/Wind Farm {parc}/"
    fd = pd.read_csv(io.BytesIO(z.read(racine + "feature_description.csv")), sep=";", encoding="latin-1")
    t = fd[fd.description.str.contains("emperature", na=False)
           & ~fd.description.str.contains("mbient|utside|xterior", na=False)]
    S = [f"{n}_avg" for n in t.sensor_name if "average" in str(t.loc[t.sensor_name == n, "statistics_type"].iloc[0])]
    ev = pd.read_csv(io.BytesIO(z.read(racine + "event_info.csv")), sep=";")
    return S, ev


def charger(parc, eid, S, W):
    z = zipfile.ZipFile(ARCHIVE)
    with z.open(f"CARE_To_Compare/Wind Farm {parc}/datasets/{eid}.csv") as fh:
        d = pd.read_csv(fh, sep=";", usecols=lambda c: c in ["id", "train_test", "status_type_id"] + S + W)
    S = [c for c in S if c in d and d[c].notna().mean() > 0.5]
    return d.dropna(subset=S + W).reset_index(drop=True), S


def scoreurs(tr, S, W):
    mu, sd = tr[W].mean().values, tr[W].std().values + 1e-12
    poly = PolynomialFeatures(2, include_bias=False)
    P = poly.fit_transform((tr[W].values - mu) / sd)
    reg = LinearRegression().fit(P, tr[S].values)
    sr = (tr[S].values - reg.predict(P)).std(0) + 1e-12
    ms, ss = tr[S].mean().values, tr[S].std().values + 1e-12
    pca = PCA(0.95, svd_solver="full").fit((tr[S].values - ms) / ss)

    def ctx(d):
        r = (d[S].values - reg.predict(poly.transform((d[W].values - mu) / sd))) / sr
        return np.sqrt(np.mean(r ** 2, 1))

    def sans_ctx(d):
        X = (d[S].values - ms) / ss
        return np.mean((X - pca.inverse_transform(pca.transform(X))) ** 2, 1)
    return {"ctx": ctx, "sans_ctx": sans_ctx}


def journalier(d, s, note):
    j = (d.id.values - d.id.values.min()) // PAR_JOUR
    df = pd.DataFrame(dict(jour=j, s=np.where(note, s, np.nan), n=note.astype(int),
                           anormal=~d.status_type_id.isin(NORMAL).values, id=d.id.values))
    g = df.groupby("jour").agg(s=("s", "mean"), n=("n", "sum"), anormal=("anormal", "any"),
                               id0=("id", "min"), id1=("id", "max"))
    g.loc[g.n < MIN_JOUR, "s"] = np.nan
    return g


def tache(args):
    parc, eid, label, debut, fin, S, W = args
    d, S = charger(parc, eid, S, W)
    tr_all = d[d.train_test == "train"]
    pr = d[d.train_test != "train"]
    rows = []
    for t_cond in ("tout", "journal"):
        tr = tr_all if t_cond == "tout" else tr_all[tr_all.status_type_id.isin(NORMAL)]
        if len(tr) < 1000 or not S:
            continue
        for nom, f in scoreurs(tr, S, W).items():
            for p_cond in ("tout", "journal"):
                note_tr = np.ones(len(tr_all), bool) if p_cond == "tout" else tr_all.status_type_id.isin(NORMAL).values
                gt = journalier(tr_all, f(tr_all), note_tr)
                seuil = np.nanquantile(gt.s, 0.99)
                note = np.ones(len(pr), bool) if p_cond == "tout" else pr.status_type_id.isin(NORMAL).values
                g = journalier(pr, f(pr), note).dropna(subset=["s"])
                g["alarme"] = g.s > seuil
                g["explique"] = g.anormal | g.anormal.shift(1, fill_value=False)
                dans = (g.id1 >= debut) & (g.id0 <= fin)
                fa = g if label == "normal" else g.iloc[:0]          # fausses alarmes : jeux normaux seulement
                fa_al = fa[fa.alarme]
                ev_al = g[dans & g.alarme]
                rows.append(dict(parc=parc, event_id=eid, label=label, capteurs=len(S), scoreur=nom,
                                 entrainement=t_cond, inference=p_cond, jours_hors=len(fa),
                                 fausses_alarmes=fa.alarme.mean() if len(fa) else np.nan,
                                 part_expliquee=fa_al.explique.mean() if len(fa_al) else np.nan,
                                 detecte=bool(len(ev_al)) if label == "anomaly" else np.nan,
                                 masque=bool(len(ev_al) and ev_al.explique.all()) if label == "anomaly" else np.nan,
                                 part_jours_anormaux=g.anormal.mean(),
                                 max_rapport=float(np.nanmax(g.s / seuil)) if len(g) else np.nan))
    return rows


def main():
    z = zipfile.ZipFile(ARCHIVE)
    T = []
    for parc in "ABC":
        S, ev = meta(z, parc)
        for r in ev.itertuples():
            T.append((parc, r.event_id, r.event_label, r.event_start_id, r.event_end_id, S, W_PARC[parc]))
    seul = os.environ.get("VB64_SEUL")
    if seul:
        T = [t for t in T if t[0] == seul][:2]
    with Pool(int(os.environ.get("VB64_PROC", "6"))) as pool:
        R = pd.DataFrame([r for rr in pool.imap_unordered(tache, T) for r in rr])
    suf = f"_essai_{seul}" if seul else ""
    R.to_csv(OUT / f"vB64_jeux{suf}.csv", index=False)
    k = ["scoreur", "entrainement", "inference"]
    a = R.groupby(k).agg(fausses_alarmes_med=("fausses_alarmes", "median"), fausses_alarmes_moy=("fausses_alarmes", "mean"),
                         detection=("detecte", "mean"), jeux=("event_id", "nunique"))
    tri = R[(R.entrainement == "journal") & (R.inference == "tout")].groupby("scoreur").agg(
        part_expliquee=("part_expliquee", "mean"), masquees=("masque", "mean"))
    from sklearn.metrics import roc_auc_score
    auc = []
    for (sc, te, inf, parc), g in R.groupby(k + ["parc"]):
        g = g.dropna(subset=["max_rapport"])
        y = (g.label == "anomaly").values
        auc.append(dict(scoreur=sc, entrainement=te, inference=inf, parc=parc, jeux_notes=len(g),
                        auroc=roc_auc_score(y, g.max_rapport) if 0 < y.sum() < len(y) else np.nan))
    auc = pd.DataFrame(auc).pivot_table(index=k, columns="parc", values=["auroc", "jeux_notes"])
    pd.set_option("display.width", 250)
    txt = (a.round(3).to_string() + "\n\nTri (entraînement journal, inférence tout)\n" + tri.round(3).to_string()
           + "\n\nAUROC sans seuil (pannes contre normaux, score journalier maximal / seuil), par parc\n"
           + auc.round(3).to_string())
    print(txt)
    (OUT / f"vB64{suf}.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
