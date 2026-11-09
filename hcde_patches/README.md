# hcde_patches/ — nos modifications au dépôt officiel H-CDE

Le clone [EPFL-IMOS/hcde](https://github.com/EPFL-IMOS/hcde) n'est **pas** versionné ici (261 Mo de
données + modèles + son propre `.git`). Ce dossier conserve tout ce qu'on y a ajouté, pour que notre
travail H-CDE soit **reproductible sans le clone**.

## Reproduire

1. `git clone https://github.com/EPFL-IMOS/hcde` dans `Probleme_Shift/hcde` (commit de base : **`b5cfa1e` « add license »**).
2. Appliquer les fichiers de ce dossier (mêmes chemins relatifs) :
   - `src/model/__init__.py` — **créé** (absent du dépôt : `GRAPH_MODELS` indéfini). Voir en-tête du fichier.
   - `src/train/__init__.py` — **créé** (absent : `create_trainer` manquant). Voir en-tête.
   - `src/model/factory.py.diff` — **patch** : enregistre la clé de config `ls-ncde` → `HCDE`. `git apply src/model/factory.py.diff` (ou report manuel, 1 ligne + commentaire).
   - `run/configs/ncmapss/hcde_{smoke,full,s2}.yaml` — nos configs (`dataset.type: windowed`, `val_split: 0.2`, `auto_resume: false`).
3. Générer le CSV d'entrée : `python Probleme_Shift/code/nc_h5_to_csv.py` (lit `Datasets/N-CMAPSS/N-CMAPSS_DS01.h5`).

**Aucun de ces changements ne touche la méthode** — packaging et config uniquement (justifications en
en-tête de chaque fichier). Détail et statut de reproduction : [../JOURNAL.md](../JOURNAL.md) (12 et
14 sept.) et [../BIBLIO.md](../BIBLIO.md) (§C, H-CDE).

## Note importante (14 sept.)

Les configs présentes tournent à un **sous-échantillonnage inadéquat** (`code/nc_h5_to_csv.py`
`STRIDE=2/4`). La recette du papier est **0,1 Hz = STRIDE=10** ; une config `hcde_s10.yaml` reste à
produire. Voir [../RESULTATS.md](../RESULTATS.md) (H-CDE) et [../DECISIONS.md](../DECISIONS.md) (D-08b).

## Configs des expériences du 15 au 21 sept. (`run/configs/ncmapss/`)

| fichiers | expérience | voir |
|---|---|---|
| `camp_DS*.yaml` | campagne multi-DS (3 unités d'entraînement par DS ; `_dNN` = dilation adaptée aux vols longs) | JOURNAL 15–17 sept. |
| `A_officiel.yaml` | config officielle du dépôt + 3 corrections minimales (type, `val_split`, `auto_resume`) | JOURNAL 18 sept. |
| `B_<scaler>_lr<lr>_s<graine>.yaml` | ablation 2×2 scaler × lr, 80 epochs sans arrêt précoce | JOURNAL 18 sept. |
| `C_standard_lr0.001_ES_s*.yaml` | arrêt précoce officiel réactivé (3 threads — voir la réserve D-17) | JOURNAL 21 sept. |
| `D_ES_s*.yaml`, `D_noES_s*.yaml` | 10 graines avec / sans arrêt précoce, 1 thread | JOURNAL 21 sept. |

**Recette de référence (D-18)** : `standard`, lr 1e-3, 80 epochs, sans arrêt précoce, lancée avec
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (le nombre de threads change le résultat — D-17). La note
« STRIDE » ci-dessus est dépassée : la recette 0,1 Hz (STRIDE=10) est appliquée depuis le 14 sept.
