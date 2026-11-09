"""Campagne multi-DS / multi-graines — protocole IDENTIQUE partout.

BUT (Martin, 15 sept.) : obtenir la VUE GLOBALE des résultats avant toute analyse et toute
interprétation. On ne conclut rien ici ; on produit la matrice de résultats.

PROTOCOLE (identique sur chaque DS)
-----------------------------------
N-CMAPSS attribue à chaque unité UNE classe de vol Fc (1=courts, 2=moyens, 3=longs). On entraîne
H-CDE sur **3 unités d'une même classe** (la classe « vue »), puis on évalue :
  - `ref`    : une unité de la MÊME classe, jamais entraînée → référence « conditions habituelles » ;
  - `unseen` : des unités d'une classe de vol JAMAIS VUE → covariate shift franc.
3 unités d'entraînement partout = runs comparables entre DS et coût maîtrisé.

FENÊTRE LENTE EN NOMBRE DE VOLS (correction du 16 sept.)
--------------------------------------------------------
La fenêtre lente vaut n_windows × dilation lignes (100 × 20 = 2000). Sur Fc1 (~435 lignes/vol) elle
couvre ~4,6 vols ; sur Fc3 (~1150–1470 lignes/vol) seulement ~1,4–1,7 vol — le chemin lent ne voit
alors pas *entre* les vols (le papier vise « roughly three flight cycles »). Pour les DS entraînés sur
Fc3, on adapte donc `dilation` afin d'égaliser les vols/fenêtre (~4,6) : clés `DS02_d53`, etc.
Le pas lent du modèle (epsilon = fast_dilation/dilation) se recalcule automatiquement (factory.py).

DS08d est EXCLU : fichier .h5 tronqué (illisible, vérifié 15 sept.).
DS04 et DS08c n'ont qu'une seule classe non vue disponible (composition du dataset).
DS08c n'a aucune unité test de la classe vue → la référence est une unité dev retenue (non entraînée).
L'horloge n'est pas une baseline (décision Martin) : aucun modèle n'a le temps en entrée.

Usage :
    python campagne_ds.py configs [--keys ...]
    python campagne_ds.py run --jobs DS02_d53:0:3 DS03:1:2 ...   # clé:graine_départ:nb_graines
    python campagne_ds.py etat
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import yaml

HCDE = Path(__file__).resolve().parents[1] / "hcde"
BASE_CFG = HCDE / "run/configs/ncmapss/hcde_lr.yaml"   # config reproduite (scaler standard, lr 1e-3)
CFG_DIR = HCDE / "run/configs/ncmapss"
PY = "/opt/anaconda3/envs/torch/bin/python"
N_PARALLEL = 4
THREADS = 2             # 10 cœurs : ~5 processus × 2 threads

# DS -> (classe vue Fc, unités d'entraînement, unité référence même classe, {unité non vue: Fc})
PLAN = {
    "DS01":  (1, [1, 4, 7],  9,  {8: 2, 10: 3}),
    "DS02":  (3, [2, 5, 10], 11, {14: 1, 15: 2}),
    "DS03":  (1, [1, 5, 9],  12, {15: 2, 10: 3}),
    "DS04":  (3, [2, 4, 5],  10, {7: 2}),                 # pas d'unité Fc1 dans DS04
    "DS05":  (1, [4, 5, 8],  10, {7: 2, 9: 3}),
    "DS06":  (1, [4, 5, 8],  10, {7: 2, 9: 3}),
    "DS07":  (1, [4, 5, 8],  10, {7: 2, 9: 3}),
    "DS08a": (1, [1, 3, 9],  10, {11: 2, 12: 3}),
    "DS08c": (3, [1, 2, 3],  4,  {7: 2}),                 # réf = unité dev retenue (pas de test Fc3)
}

DIL_REF = 20            # dilation du papier (Fc1, ~435 lignes/vol → ~4,6 vols/fenêtre)
RPF_REF = 435           # lignes par vol, Fc1 (mesuré)
# dilation adaptée = round(20 × lignes_par_vol / 435), mesuré sur les unités d'entraînement
ADAPT = {"DS02": 53, "DS04": 66, "DS08c": 68}


def key_of(ds: str, dil: int = DIL_REF) -> str:
    return ds if dil == DIL_REF else f"{ds}_d{dil}"


def parse_key(key: str) -> tuple[str, int]:
    if "_d" in key:
        ds, d = key.rsplit("_d", 1)
        return ds, int(d)
    return key, DIL_REF


KEYS = list(PLAN) + [key_of(ds, d) for ds, d in ADAPT.items()]


def cfg_path(key: str) -> Path:
    return CFG_DIR / f"camp_{key}.yaml"


def make_configs(keys=None):
    base = yaml.safe_load(BASE_CFG.read_text())
    for key in keys or KEYS:
        ds, dil = parse_key(key)
        fc, train, ref, unseen = PLAN[ds]
        c = dict(base)
        c["out_dir"] = f"results/camp_{key}"
        c["dataset"] = dict(base["dataset"])
        c["dataset"]["train_files"] = [f"N-CMAPSS_{ds}.csv"]
        c["dataset"]["test_files"] = [f"N-CMAPSS_{ds}.csv"]
        c["dataset"]["train_units"] = train
        c["dataset"]["dilation"] = dil
        # Sélection du modèle sur la référence MÊME CLASSE : les unités non vues ne sont
        # jamais regardées pendant l'entraînement (sinon la mesure covariate serait biaisée).
        c["dataset"]["test_units"] = [ref]
        cfg_path(key).write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
        print(f"{cfg_path(key).name}: train Fc{fc} {train} | réf {ref} | non vues {unseen} | dilation {dil}")


def set_seed(key: str, start: int):
    """main.py lit `cfg.seed` une fois au démarrage puis incrémente : on fixe la graine de départ
    JUSTE AVANT le lancement, pour ne jamais réécraser des graines déjà calculées."""
    p = cfg_path(key)
    c = yaml.safe_load(p.read_text())
    c["seed"] = start
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))


def run(jobs, parallel: int = N_PARALLEL):
    """jobs = [(clé, graine_départ, nb_graines)], lancés dans l'ordre. Un même clé n'apparaît
    qu'une fois (pas deux processus sur le même dossier de résultats)."""
    jobs = list(jobs)
    running, done = [], []
    env = {**os.environ, "PYTHONPATH": ".", "OMP_NUM_THREADS": str(THREADS),
           "MKL_NUM_THREADS": str(THREADS)}
    while jobs or running:
        while jobs and len(running) < parallel:
            key, start, n = jobs.pop(0)
            set_seed(key, start)
            log = Path(f"/tmp/camp_{key}_s{start}.log")
            p = subprocess.Popen(
                [PY, "run/main.py", "--cfg", str(cfg_path(key)), "--repeat", str(n)],
                cwd=HCDE, stdout=log.open("w"), stderr=subprocess.STDOUT, env=env)
            running.append((key, p))
            print(f"[lancé] {key} graines {start}..{start + n - 1} (pid {p.pid})", flush=True)
            time.sleep(5)      # laisser le processus lire sa config avant d'en réécrire une autre
        time.sleep(30)
        for key, p in list(running):
            if p.poll() is not None:
                running.remove((key, p)); done.append(key)
                print(f"[fini ] {key} code={p.returncode}", flush=True)
    print("campagne terminée :", done, flush=True)


def etat():
    for key in KEYS:
        out = HCDE / f"run/results/camp_{key}/camp_{key}"
        seeds = sorted(p.name for p in out.glob("*") if p.name.isdigit()) if out.exists() else []
        fini = [s for s in seeds if any((out / s / "ckpt").glob("79.ckpt"))]
        print(f"{key:11s} dossiers={seeds or '—'}  terminés={fini or '—'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["configs", "run", "etat"])
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--jobs", nargs="*", default=[], help="clé:graine_départ:nb_graines")
    ap.add_argument("--parallel", type=int, default=N_PARALLEL)
    a = ap.parse_args()
    if a.cmd == "configs":
        make_configs(a.keys)
    elif a.cmd == "run":
        run([(j.split(":")[0], int(j.split(":")[1]), int(j.split(":")[2])) for j in a.jobs],
            a.parallel)
    else:
        etat()
