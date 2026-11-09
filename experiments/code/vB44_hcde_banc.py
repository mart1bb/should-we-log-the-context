"""vB44 — H-CDE dans le banc, sur les 12 cas N-CMAPSS de vB35/vB39 (R13 de PAPIER_ICDE § 8).

Relecture (W2) : H-CDE, méthode avec contexte de l'état de l'art utilisée en motivation (Fig. 1), est absente du
tableau III. Décision de Martin (13 oct.) : H-CDE sur N-CMAPSS, exclusion justifiée ailleurs (H-CDE modélise la
dynamique rapide À L'INTÉRIEUR d'un vol ; C-MAPSS, roulements, gaz n'ont qu'une ligne par vol).

Protocole déclaré :
  - recette de référence D-18 (`hcde/run/configs/ncmapss/hcde_lr.yaml` : scaler standard, lr 1e-3, 80 epochs, sans
    arrêt précoce), 1 thread par run (D-17), dilation 20 (DS01, DS05, DS07) ; 3 graines (0, 1, 2) ;
  - mêmes 12 cas que vB39 : par DS, chaque unité Fc1 tour à tour en référence, entraînement = les autres Fc1, test =
    toutes les unités non vues (Fc2, Fc3) ; `test_units = [réf]` dans la config (comme campagne v3) ;
  - les 3 cas identiques à la campagne v3 (DS01 réf 9, DS05 réf 10, DS07 réf 10) réutilisent ses runs c3 graines 0–2
    (même config de base) ; 27 nouveaux runs, configs `vB44_*` dans `hcde/run/configs/ncmapss/` (préfixe vB_ ; aucun
    fichier existant modifié) ;
  - indicateur SANS ÉTIQUETTES : PC1 du latent lent (dernier pas, 5 dimensions ; ACP à 3 composantes ajustée sur les
    fenêtres d'entraînement, comme it12), orienté pour croître avec le rang du vol sur les unités d'entraînement
    (règle de toutes nos méthodes). AUCUNE porte de validité (la porte de c3 utilise la vérité terrain). Rapporté à
    part, comme diagnostic seulement : l'orientation d'it12 (par la vérité terrain d'entraînement) ;
  - mesure : évaluateur commun, Spearman par vol avec la dégradation, unités non vues ; par cas, médiane des unités ;
    graines agrégées comme vB35 (médiane des graines des médianes par cas).
Lecture fixée AVANT de mesurer : « la méthode avec contexte de l'état de l'art fait mieux que l'a priori temporel sur
N-CMAPSS » si sa médiane ≥ meilleure sans contexte (vB39, médiane du max par cas) + 0,05. Attendu (déclaré) : non
(sa réponse à la dégradation se contracte hors support, Fig. 1).

Machine : choisie par Martin. Usage :
    python vB44_hcde_banc.py configs   # écrit les 27 configs et la liste des runs
    python vB44_hcde_banc.py run       # N_PAR runs en parallèle, 1 thread chacun
    python vB44_hcde_banc.py etat
    python vB44_hcde_banc.py eval      # → results/vB44_par_unite.csv, vB44.txt
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ICI = Path(__file__).resolve().parent
TRONC = ICI.parent.parent / "code"
sys.path.insert(0, str(ICI))
sys.path.insert(0, str(TRONC))
import vB01_sonde_marge as v  # noqa: E402

HCDE = ICI.parent.parent / "hcde"
BASE_CFG = HCDE / "run/configs/ncmapss/hcde_lr.yaml"
CFG_DIR = HCDE / "run/configs/ncmapss"
PY = "/opt/anaconda3/envs/torch/bin/python"
DS_, DIL, GRAINES = ["DS01", "DS05", "DS07"], 20, (0, 1, 2)
C3 = {"DS01": 9, "DS05": 10, "DS07": 10}                  # réf. de la campagne v3 (même découpage)
JOBS = ICI.parent / "results" / "vB44_jobs.txt"
N_PAR = int(os.environ.get("VB44_PAR", "12"))


def plan():
    out = []
    for ds in DS_:
        fc = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv", usecols=["unit", "Fc"]).groupby("unit").Fc.first()
        fc1 = sorted(fc[fc == 1].index)
        unseen = sorted(fc[fc != 1].index)
        for ref in fc1:
            out.append((ds, ref, [u for u in fc1 if u != ref], unseen))
    return out


def nom(ds, ref, g):
    return f"c3_{ds}_s{g}" if C3[ds] == ref else f"vB44_{ds}_r{ref}_s{g}"


def run_dir(n, g):
    return HCDE / f"run/results/{n}/{n}/{g}"


def configs():
    base = yaml.safe_load(BASE_CFG.read_text())
    noms = []
    for ds, ref, train, _ in plan():
        for g in GRAINES:
            n = nom(ds, ref, g)
            if n.startswith("c3_"):
                continue
            c = yaml.safe_load(yaml.safe_dump(base))
            c["seed"] = g
            c["out_dir"] = f"results/{n}"
            c["dataset"]["train_files"] = [f"N-CMAPSS_{ds}.csv"]
            c["dataset"]["test_files"] = [f"N-CMAPSS_{ds}.csv"]
            c["dataset"]["train_units"] = [int(u) for u in train]
            c["dataset"]["test_units"] = [int(ref)]
            c["dataset"]["dilation"] = DIL
            (CFG_DIR / f"{n}.yaml").write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
            noms.append(n)
    JOBS.write_text("\n".join(noms) + "\n")
    print(f"{len(noms)} runs -> {JOBS}")


def run():
    env = {**os.environ, "PYTHONPATH": ".", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    cmd = (f"xargs -P {N_PAR} -I{{}} sh -c "
           f"'{PY} run/main.py --cfg run/configs/ncmapss/{{}}.yaml --repeat 1 > run/results/{{}}.log 2>&1' "
           f"< {JOBS}")
    subprocess.run(cmd, shell=True, cwd=HCDE, env=env)


def fini(n, g):
    return (run_dir(n, g) / "ckpt/79.ckpt").exists()


def etat():
    for ds, ref, _, _ in plan():
        print(ds, f"réf {ref}", [g for g in GRAINES if fini(nom(ds, ref, g), g)])


def evaluer_run(ds, ref, train, unseen, g):
    import torch
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import it10_hcde_covariate as it10
    import it12_campagne_eval as it12
    from evaluateur import evaluer
    n = nom(ds, ref, g)
    assert fini(n, g), f"{n} : entraînement non terminé"
    it10.DILATION = DIL
    v.it10.DILATION = DIL
    df = pd.read_csv(v.DATASETS / f"N-CMAPSS_{ds}.csv")
    scaler = StandardScaler().fit(df[df.unit.isin(train)][it10.FEATS].values)
    model = it12.load_model(run_dir(n, g) / "model.pt")
    torch.set_num_threads(1)
    fen, lat = [], []
    for u in list(train) + [ref] + list(unseen):
        f = v.fenetres(df, u)
        _, l = it10.infer(model, scaler, df, u)
        assert len(l) == len(f), f"{n} u{u} : {len(l)} latents pour {len(f)} fenêtres"
        fen.append(f.assign(role="train" if u in train else "test"))
        lat.append(l)
    fen, lat = pd.concat(fen, ignore_index=True), np.vstack(lat)
    tr = (fen.role == "train").values
    pc = PCA(n_components=3).fit(lat[tr]).transform(lat)[:, 0]
    rang = fen[tr].groupby("unit").cycle.rank(pct=True).values
    sans = pc if np.corrcoef(pc[tr], rang)[0, 1] >= 0 else -pc
    meta = pd.concat([it12.window_meta(df, u) for u in list(train)], ignore_index=True)
    verite = pc if np.corrcoef(pc[tr], meta.degradation.values)[0, 1] >= 0 else -pc
    rows = []
    for meth, s in [("H-CDE", sans), ("H-CDE_orientation_verite", verite)]:
        res = evaluer(fen.drop(columns="role").assign(score=s), ds=ds, train_units=train, ref_unit=ref,
                      non_vues=unseen)
        # 15 oct. : on garde aussi l'unité de référence (classe vue) pour la Fig. 1 (vu / non vu, en Spearman)
        rows += [dict(jeu="N-CMAPSS", cas=f"{ds} aucune ref u{ref}", methode=meth, graine=g, unite=str(r.unite),
                      role=r.role, rho=float(r.corr_spearman), AUROC=float(r.AUROC), reponse=float(r.reponse))
                 for r in res.itertuples()]
    return rows


def evaluation():
    rows = []
    for ds, ref, train, unseen in plan():
        for g in GRAINES:
            rows += evaluer_run(ds, ref, train, unseen, g)
            print(ds, ref, g, "évalué", flush=True)
    L = pd.DataFrame(rows)
    L.to_csv(ICI.parent / "results" / "vB44_par_unite_avec_ref.csv", index=False)
    L = L[L.role == "non_vue"].drop(columns=["role", "AUROC", "reponse"])
    L.to_csv(ICI.parent / "results" / "vB44_par_unite.csv", index=False)
    c = L.groupby(["cas", "methode", "graine"]).rho.median().reset_index()
    C = c.groupby(["cas", "methode"]).rho.median().unstack("methode")
    ref = pd.read_csv(ICI.parent / "results" / "vB39_cas.csv")
    ref = ref[ref.jeu == "N-CMAPSS"].set_index("cas")
    C["meilleure sans contexte"] = ref[["lenteur", "tendance_moy", "tendance_reg"]].max(axis=1)
    C["lenteur"] = ref.lenteur
    with open(ICI.parent / "results" / "vB44.txt", "w") as fh:
        for x in ["=== H-CDE sur les 12 cas N-CMAPSS (médiane des graines) ===", C.round(3).to_string(),
                  "\nmédianes : " + C.median().round(3).to_string().replace("\n", " ; ")]:
            print(x)
            print(x, file=fh)
        m = C.median()
        verdict = m["H-CDE"] >= m["meilleure sans contexte"] + 0.05
        x = (f"lecture fixée : H-CDE {m['H-CDE']:.2f} contre meilleure sans contexte {m['meilleure sans contexte']:.2f}"
             f" → {'fait mieux' if verdict else 'ne fait pas mieux'}")
        print(x)
        print(x, file=fh)


if __name__ == "__main__":
    {"configs": configs, "run": run, "etat": etat, "eval": evaluation}[sys.argv[1]]()
