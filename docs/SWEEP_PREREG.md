# Checkpoint 3 pre-registration: sensitivity sweeps for FedMiX

Written and committed **before any run**, on 2026-11-03. The principal approved it interactively: sensitivity sweeps; method name **FedMiX**.

## Method
**FedMiX** (`fedmix`) is SFS-PCL-V without PCL, i.e. the same computation as `sfspcl_v_ccvr_ce`. It consists of:
- SFS;
- class-conditional Gaussian-mixture virtual samples (r = 1.0);
- validation-guarded CCVR;
- plain cross-entropy.

It was adopted at checkpoint 2c by the pre-registered rule: on seeds 15-19 it beat SFS-PCL-V by +2.29 pp and SCAFFOLD by +4.80 pp, with p_Holm 0.020 in that suite.

## Design (`sweep_main`)
The sweeps vary one factor at a time around the main setting (K = 20, alpha = 0.1, rho = 0.5, T = 50):

| Factor | Values |
|---|---|
| K | {10, 50, 100} |
| alpha | {0.05, 0.5, 1.0} |
| rho (participation) | {0.2, 1.0} |
| Main setting (repeated) | K = 20, alpha = 0.1, rho = 0.5 |

- **Methods:** FedMiX, SCAFFOLD, FedDyn, CCVR and SFS-PCL.
- **FedMiX strength:** r in {0.25, 0.5, 2.0}.
- **Seeds:** **20-24**, fresh.
- **Data:** CICIoT2023, Edge-IIoTset and TON_IoT, fine labels only (grouped labels are omitted for cost).
- **Metrics:** the standard metric and the site-shuffled metric for every run.
- 720 runs in total.

## Hyperparameters
- No new tuning.
- FedDyn, CCVR, FedAvgM and FedLC use their checkpoint-2b re-tuned values (`configs/tuned.yaml`).
- The values used by `grid_main` are preserved in `configs/tuned_grid_main.yaml`.

## Reporting (`analysis/sweep_rederive.py`)
- For each setting, method and dataset: mean ± sd of the standard macro-F1, and the site-shuffled mean.
- **Paired tests:** FedMiX minus each baseline over the 15 (dataset, seed) cells per setting, with two-sided Wilcoxon.
- **Holm correction** over all (setting, baseline) pairs, done separately for the standard and the site-shuffled metric.
- **FedMiX's r curve.**
- These are descriptive robustness results: no go/no-go rule. Every setting is reported, including any where FedMiX loses.

## Operations
- One L4 pod, fully autonomous, with progress snapshots every 90 min.
- Results are pushed to branch `sfspcl-v2-sweep`.
- The hard stop is 18 h.
- Estimated cost: about $7-9 (total about $21-23 of the $30 cap).

## Addendum (2026-11-03 ~08:00 UTC, before any sweep run finished)
- **What happened:** the mixture pre-build stopped the job because the Dirichlet partition could not satisfy the minimum client size (50 samples) for TON_IoT at K = 100 for two seeds. TON_IoT has 119,874 samples after deduplication.
- **Why it doesn't bias the comparison:** the partition is the same for every method, so these cells are infeasible for all methods alike.
- **Fix:** the job now logs this and continues. Infeasible cells are reported as missing, and paired tests use the remaining cells.
- **Unchanged:** no other change to the protocol.
- **Cost:** the stop left the pod idle for about 1.5 h, roughly $0.9.
