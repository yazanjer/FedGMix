# Checkpoint 2b pre-registration: site-shuffled diagnostic and baseline grid extension

Written and committed **before any run**, on 2026-11-02. The principal approved it interactively:
- diagnose the local-normalisation anomaly first;
- run a disclosed extended-grid check for the edge-selected baselines;
- decide about any local-normalisation variant only after the diagnostic.

## 1. Site-shuffled diagnostic (`diag_shift`)

**Observation in `grid_main`:** methods with client-local input normalisation score far above the centralised reference:
- SFS-PCL-V without SFS (`sfspcl_v_ccvr_ln`): 76.4 vs 62.6 on CICIoT2023 fine, 95.4 vs 88.7 on Edge-IIoTset fine.
- CCVR, FedBN, and SCAFFOLD vs SCAFFOLD+SFS show the same pattern.

**Hypothesis:** a local scaler makes inputs site-specific. Because each site's test traffic follows the same skewed label mix as its training traffic, the model can use site identity as a label prior. This would not hold when a site receives a different attack mix, which is the situation an IDS must handle.

**Test:**
- Re-run, with unchanged training, the cells of 12 methods: sfspcl_v_ccvr, sfspcl_v_ccvr_ln, sfspcl_v, sfspcl, ccvr, scaffold, scaffold_sfs, feddyn, fedavg, statavg, fedbn and centralised.
- Cells: 3 datasets, fine labels, seeds 10-12 (108 runs).
- The run_ids and standard test scores must equal those in `grid_main`. This doubles as a reproducibility check.
- Additional metric `test_site_shuffled`: pool every client's test samples and assign each to a uniformly random site (seeded). That site normalises it with its own scaler (and its own BN statistics for FedBN). The final model classifies it, and macro-F1 is taken over all samples.
- For global-normalisation methods (except FedBN), the inputs are unchanged by construction. This is the control.

**Reading:**
- The **site-identity effect** of a method is (standard macro-F1 − site-shuffled macro-F1).
- If local-normalisation methods lose much more than global-normalisation methods, and `sfspcl_v_ccvr_ln` falls to or below `sfspcl_v_ccvr` when shuffled, the gain from local normalisation is a site-identity artefact. It is then reported as such and is not used to choose a method.
- Either way, the site-shuffled scores of every method, including SCAFFOLD, are reported as a robustness result.

## 2. Extended tuning of edge-selected baselines (`tune_ext`, `grid_ext`)

**Why:** in `tune_grid`, CCVR (lr 0.3), FedDyn (alpha 0.1, lr 0.02), FedAvgM (beta 0.3) and FedLC (tau 0.01) were selected on a grid edge.

**Extended grids** (the previously selected edge value is included, same tuning protocol):

| Baseline | Extended grid |
|---|---|
| CCVR | lr {0.3, 0.6, 1.0} |
| FedDyn | alpha {0.1, 0.3, 1.0} x lr {0.005, 0.01, 0.02} |
| FedAvgM | beta {0.1, 0.2, 0.3} |
| FedLC | tau {0.001, 0.003, 0.01} |

- **Re-runs:** every baseline whose selection changes (and FedLC+SFS if FedLC changes) is re-run on all `grid_main` cells.
- **Reporting:** this is a **supplementary sensitivity table**. The pre-registered main table and the tier decision of checkpoint 2 are not changed.

## Operations
- One L4 pod, torch 2.11.0+cu128.
- Results are pushed to branch `sfspcl-v2-diag`.
- Estimated cost: about $2-3.
