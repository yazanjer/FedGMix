# Full-grid pre-registration (SFS-PCL v2, checkpoint 2)

Written and committed **before any grid run**, on 2026-11-02. The principal approved the plan interactively:
- Scope: main comparison + ablation first, with a checkpoint before the sweeps.
- Baselines: the full list, including FedDyn.
- TON_IoT is held out from tuning.
- Seeds 10-14, fresh.
- GPU: wait for L4.
- Name: **SFS-PCL-V**.

## Proposed method
SFS-PCL-V (`sfspcl_v_ccvr`) consists of:
- SFS;
- PCL (tau 1);
- class-conditional Gaussian-mixture virtual samples, with r = 1.0, at most 8 components per client and class, and at least 50 samples per shared component;
- validation-guarded CCVR, with ccvr_lr 0.1.

It passed the pilot-3 go rule on seeds 5-6. Every earlier candidate (C1-C4, C3+C1a, P1-P4, V) is disclosed.

## Data
| Dataset | Source | Tuning |
|---|---|---|
| CICIoT2023 | prepared sha256 dcad9db3... | used for tuning |
| Edge-IIoTset | prepared sha256 9be2aed2... | used for tuning |
| TON_IoT (network) | Kaggle mirror of the UNSW Train_Test_Network.csv; prepared sha256 c364aa93... | **held out from all tuning** |

Labels: fine and grouped.

## Setting
- K = 20, alpha = 0.1, rho = 0.5, T = 50.
- One GPU type for every run (NVIDIA L4), torch 2.11.0+cu128.

## Methods (24)
- **Proposed:** sfspcl_v_ccvr.
- **Ablation:**
  - sfspcl_v (no CCVR)
  - sfspcl_ccvrg (no virtual samples)
  - sfspcl_v_ccvr_ce (no PCL)
  - sfspcl_v_ccvr_ln (no SFS)
  - sfspcl (no V, no CCVR)
- **Baselines:**
  - scaffold
  - feddyn (new, stateful)
  - fedavg, fedprox, fedlc, fedrs, statavg
  - fedavgm, feddecorr, fedetf, ccvr (new)
  - fedprox_sfs, scaffold_sfs, fedlc_sfs, fedrs_sfs
- **References, not tested:** fedbn, fedbn_sfs, centralised.

## Tuning (`tune_grid`)
- Same protocol as every earlier method: seed 100, fine labels, mean validation macro-F1 over CICIoT2023 and Edge-IIoTset.
- The new baselines are tuned on their own hyperparameter:

| Method | Grid |
|---|---|
| FedAvgM | beta {0.3, 0.5, 0.7, 0.9} |
| FedDecorr | {0.01, 0.1, 1, 3, 10} |
| FedETF | scale {2, 5, 10, 20, 40} |
| CCVR | lr {0.01, 0.03, 0.1, 0.3} |
| FedDyn | alpha {0.001, 0.01, 0.1} x client lr {0.02, 0.05, 0.1} |

- **Fairness fix 2:** FedLC tau {0.01, 0.03, 0.1} and FedRS alpha {0.9, 0.95, 1.0} extend the v1 grid edges.
- A selection that is still on an edge is reported, not extended.
- All other methods keep their earlier tuned values.

## Evaluation
- Main suite `grid_main`: 24 methods x 3 datasets x 2 label levels x seeds **10-14** = 720 runs.
- Analysis: `analysis/grid_rederive.py`, an independent re-derivation from stored files.

## Success tiers (proposed vs SCAFFOLD; from the agent brief, unchanged)
**Stretch tier**
- The mean over all 30 (dataset, labels, seed) pairs is > 0.
- The Holm-corrected two-sided Wilcoxon p is < 0.05. The Holm family is all compared methods except the references.

**Minimum tier**
- Fine labels (15 pairs): mean difference >= -1.0 pp, with the bootstrap 95% CI lower bound > -2.0 pp.
- Grouped labels (15 pairs): mean difference > 0.
- Mean upload <= 0.6 x SCAFFOLD's.

**If neither tier is met:** stop and report. There is no re-tuning, no extra seeds and no metric change.

**Also reported:**
- every baseline comparison with Holm-corrected p;
- Friedman test with Nemenyi CD;
- per-block tables;
- the ablation table;
- upload cost.

## Operations
- Results were pushed to a private results branch from the compute pod.
- A progress snapshot is pushed every 2 h.
- The pod self-terminates; the hard stop is 18 h.
- Estimated cost: about $7-9.
