# Should We Log the Context?

Code, result files and data preparation scripts for the paper

> M. Bonsergent-Brachet, J. Read, D. Abboud. *[Experiment, Analysis, and Benchmark] Should We Log the Context?
> Contextual Attributes and Event Logs for Unsupervised Degradation Tracking under Unseen Operating Conditions.*
> Submitted to ICDE 2027.

The paper compares nineteen unsupervised methods that ignore, remove, become invariant to, or condition on the
operating context, on six public datasets split so that test contexts lie outside the training support (part 1). It
then injects measurement changes, operation changes and maintenance into 51 cases of this benchmark and measures what
an event log should contain (part 2), and uses the status codes of CARE to Compare as a real event log.

The observations on an industrial robot fleet (Section II-B and Fig. 1 of the paper) use proprietary data and are not
part of this repository.

## Layout

```
code/                 shared modules: slow feature analysis (sfa.py), N-CMAPSS loading and evaluation,
                      C-MAPSS regime splits (a6_cmapss_regime_retire.py), N-CMAPSS conversion (nc_h5_to_csv.py)
experiments/code/     one script per experiment, numbered vB01 ... vB74 (only those used by the paper)
experiments/results/  outputs of these scripts (text summaries and CSV files), as used in the paper
experiments/figures/  figures of the paper generated from the results
experiments/papier/   LaTeX tables and the event-log figure generated from the results
hcde_patches/         the changes we applied to the official H-CDE repository (see below)
results/              one reference file read by vB01
data/                 the public datasets (not versioned; see "Data")
hcde/                 clone of the official H-CDE repository (not versioned; see "Setup")
```

Each script starts with a docstring that states its question, its protocol and its outputs. Comments and docstrings
are in French; the code itself (names of methods, datasets and columns) follows the paper.

## Requirements

Python 3.11 with the packages of `requirements.txt` (tested with the versions listed there). Running H-CDE itself
(`vB44`) also needs the requirements of its repository (`hcde/requirements.txt`). All experiments run on a CPU, one
thread per training (`OMP_NUM_THREADS=1`).

```bash
pip install -r requirements.txt
```

## Data

The datasets are public and are not redistributed here. Download them into `data/` (or point the environment variable
`DATA_ROOT` to another folder) with the following layout.

| Dataset | Source | Expected location under `data/` |
|---|---|---|
| N-CMAPSS (DS01, DS05, DS07) | NASA Prognostics Data Repository, *Turbofan Engine Degradation Simulation-2*; Arias Chao et al., Data 6(1):5, 2021 | `N-CMAPSS/N-CMAPSS_DS01.h5`, ... |
| C-MAPSS (FD002, FD004) | NASA Prognostics Data Repository, *Turbofan Engine Degradation Simulation*; Saxena et al., PHM 2008 | `CMAPSS/archive/train_FD002.txt`, ... |
| NASA randomized battery usage | Bole et al., PHM 2014; Zenodo, doi:10.5281/zenodo.15277374 | `NASA_Batteries_RW/<group>/<group>/data/Matlab/RW*.mat` |
| Paderborn bearings under time-varying conditions | features of github.com/alireza-javanmardi/bearing-RUL (folder `data/LDM`); Javanmardi et al., PHME 2024 | `Bearings_UPB_TimeVarying/data/LDM/{op,temp,fft}/` |
| Gas sensor array drift at different concentrations | UCI Machine Learning Repository, dataset 270; Vergara et al., 2012 | `Gas_Sensor_Drift_Concentrations/batch1.dat`, ... |
| PVDAQ | NREL, OEDI, doi:10.25984/1846021; systems 4, 10, 33, 34, 35, 50, 51, 1276, 1277, 1278 | `PVDAQ/system_id=<id>/year=*/month=*/day=*/*.csv` |
| CARE to Compare | Gück et al., 2024; Zenodo, doi:10.5281/zenodo.15846963 | `Wind_CARE_to_Compare/CARE_To_Compare.zip` (read without extraction) |

## Setup

Most scripts import the windowing and evaluation code of H-CDE (Zhao and Fink), so its repository is needed even for
the datasets other than N-CMAPSS.

1. Clone the official repository into `hcde/` and check out the commit we used:
   ```bash
   git clone https://github.com/EPFL-IMOS/hcde hcde
   git -C hcde checkout b5cfa1e
   ```
2. Apply our changes, which concern packaging and configuration only (`hcde_patches/README.md` explains each one):
   ```bash
   cp hcde_patches/src/model/__init__.py hcde/src/model/__init__.py
   cp hcde_patches/src/train/__init__.py hcde/src/train/__init__.py
   (cd hcde && git apply ../hcde_patches/src/model/factory.py.diff)
   mkdir -p hcde/run/configs/ncmapss && cp hcde_patches/run/configs/ncmapss/*.yaml hcde/run/configs/ncmapss/
   ```
3. Convert the N-CMAPSS subsets to the CSV format read by H-CDE and by our scripts (one point in ten per flight,
   0.1 Hz, as in the H-CDE paper):
   ```bash
   for ds in DS01 DS05 DS07; do python code/nc_h5_to_csv.py --ds $ds; done
   ```

## Reproducing the paper

Run the scripts from `experiments/code/`, for instance `cd experiments/code && python vB30_capteurs_gaz.py`. Each
script writes into `experiments/results/` (and caches intermediate data in `experiments/cache/`). The scripts that
only format results (`vB36`, `vB38`, `vB68`, `vB74`) read the files already present in `experiments/results/`, so the
tables and figures can be regenerated without the data.

| Paper | Scripts |
|---|---|
| Table I (methods) | methods in `code/sfa.py`, `vB08` (span_W), `vB24` (residuals), `vB26` (neural residual, latent model), `vB32` (trends), `vB33` (regime labels), `vB34` (DANN), `vB45` (V-REx) |
| Table II (datasets) and data preparation | `vB19`, `vB22`, `vB29`, `vB30`, `vB31_pvdaq_charger`, `code/a6_cmapss_regime_retire.py` |
| Table III (main results) | `vB35`, `vB39` (per unit), `vB40` (tests and bootstrap), `vB44` (H-CDE), then `vB38` (LaTeX) |
| Section IV-A (support, in-support controls, tuning) | `vB70` (distance to the support), `vB47` (in-support controls), `vB41` (oracle tuning), `vB50` (sensitivity) |
| Fig. 2 and Fig. 3 | `vB36` (from `vB35`, `vB37`); C1: `vB05`, `vB37`; C3: `vB22`, `vB72` |
| Section IV-C (which attributes to log, regime labels) | `vB42`, `vB48`, `vB33`, `vB34`, `vB45` |
| Section IV-D (method selection, alarm thresholds) | `vB24`, `vB27`, `vB29`, `vB30`, `vB31`; thresholds: `vB46` |
| Table IV (event logs) | `vB71`, then `vB74` (LaTeX); detected dates from `vB67` |
| Fig. 4 (event logs against the magnitude) | `vB67`, then `vB68` |
| Section V-B (corrections on clean data, splitting the lives, alarms) | `vB62` and `vB71` (corrections on clean data), `vB73` (splitting the lives), `vB67` then `vB68` (alarms) |
| Table V (CARE to Compare) | `vB64`, then `vB68` (LaTeX) |

## License

MIT (see `LICENSE`). The files in `hcde_patches/` modify the H-CDE repository, which is also under the MIT license
(copyright Mengjie Zhao and Olga Fink).
