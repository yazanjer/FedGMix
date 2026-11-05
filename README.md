# FedGMix

Code, configurations, pre-registration documents, per-run results and analysis scripts for the article

> A. Fahad, M. F. Alomari and Y. Aljeroudi, **FedGMix: Stateless Federated Intrusion Detection with Shared Class-Conditional Mixtures under Label Skew and Site-Level Shift** (manuscript under review).

Archived version: [https://doi.org/10.5281/zenodo.23986522](https://doi.org/10.5281/zenodo.23986522)

FedGMix is a stateless federated learning method for flow-based intrusion detection. Clients share global feature statistics and, once, a small Gaussian mixture per class; each client then trains on its own flows plus balanced virtual samples drawn from the merged mixtures, and the server re-fits the classifier head only when federated validation improves. The repository also implements the site-shuffled test, which measures how much a method relies on site identity rather than on the attack.

## Contents

| Path | Content |
|---|---|
| `src/fedsim/` | Simulator: data preparation, partitioning, client training, aggregation, mixtures, CCVR, evaluation |
| `configs/method/` | One YAML file per method (see the mapping below) |
| `configs/dataset/`, `configs/main.yaml`, `configs/tuned*.yaml` | Dataset definitions, defaults and tuned hyperparameters |
| `configs/suites/` | Experiment suites (lists of datasets, label granularities, methods and seeds) |
| `docs/*_PREREG.md` | Decision rules fixed before the corresponding runs |
| `scripts/` | Data download and preparation, suite runner, hyperparameter selection |
| `analysis/` | Scripts that re-derive every reported number from the stored per-run results |
| `results_v2/` | Per-run `results.json` files and reports (`grid/` and `sweep/` are packed as `results.tgz`) |
| `tests/` | CPU tests (determinism, resume, method components, DP accounting) |

## Method names

| Configuration | Name in the article |
|---|---|
| `fedmix` (same computation as `sfspcl_v_ccvr_ce`) | **FedGMix** |
| `sfspcl_v_ccvr` | FedGMix+PCL |
| `sfspcl_v` | FedGMix+PCL without CCVR |
| `sfspcl_ccvrg` | SFS+PCL + guarded CCVR |
| `sfspcl_v_ccvr_ln` | FedGMix+PCL (local norm.) |
| `sfspcl` | SFS+PCL (shared feature statistics + prior-calibrated loss) |
| `fedmix_fit`, `fedmix_k1`, `fedmix_novirt`, `fedmix_dp` | Corrected guard, single Gaussian, no virtual samples, DP release (Supplementary Section S4) |
| `scaffold`, `feddyn`, `ccvr`, `fedavg`, `fedprox`, `fedlc`, `fedrs`, `statavg`, `fedavgm`, `feddecorr`, `fedetf`, `fedbn`, `*_sfs`, `centralised` | Baselines as named |

Other `sfspcl_*` configurations are candidates from the design path (Supplementary Section S1).

## Seeds and suites

| Seeds | Suite | Use |
|---|---|---|
| 100 | `tune_grid`, `tune_v*` | hyperparameter tuning |
| 0-6 | `pilot_v2`, `pilot_v3`, `pilot_v4` | design-path pilots |
| 10-14 | `grid_main` | main comparison |
| 15-19 | `h2h_pcl` | confirmatory test |
| 20-24 | `sweep_main` | sensitivity study |
| 10-12 | `diag_shift` | site-shuffled test |
| 25-27 | `dp_check` | DP release and ablations |

## Reproducing

```bash
pip install torch numpy scipy pandas scikit-learn pyyaml omegaconf hydra-core kaggle
python scripts/download_data.py --dataset ciciot2023 --raw data/raw/ciciot2023   # also edgeiiot, toniot (Kaggle credentials required)
python scripts/prepare_data.py --dataset ciciot2023 --raw data/raw/ciciot2023 --out data/prepared
python scripts/run_suite.py --suite grid_main --workers 4 --out outputs_v2 --data data/prepared
python analysis/grid_rederive.py --out outputs_v2 --report grid_report.md
```

To re-derive the reported numbers without training, unpack the stored results and run the analysis scripts on them:

```bash
mkdir -p stored && tar xzf results_v2/grid/results.tgz -C stored && tar xzf results_v2/sweep/results.tgz -C stored
python analysis/grid_rederive.py --out stored/outputs_v2
```

Runs are deterministic on one GPU type (NVIDIA L4, PyTorch 2.11); identical configurations share one run identifier. Tests: `PYTHONPATH=src python tests/test_core.py` (likewise `test_v2.py` to `test_v6.py`).

## Data

CICIoT2023, Edge-IIoTset and TON_IoT are public datasets and are not redistributed here; see `configs/dataset/` for sources and preprocessing.

## Licence and citation

MIT licence (see `LICENSE`). Please cite the article and the archived version ([doi:10.5281/zenodo.23986522](https://doi.org/10.5281/zenodo.23986522)); `CITATION.cff` contains the metadata.
