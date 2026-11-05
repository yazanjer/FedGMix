# Checkpoint 2c pre-registration: SFS-PCL-V with vs without PCL

Written and committed **before any run**, on 2026-11-03. The principal approved it interactively ("Test the no-PCL version first"; framing: two-part claim).

## Why
- In the pre-registered ablation of `grid_main` (seeds 10-14), `sfspcl_v_ccvr_ce` (SFS-PCL-V with plain cross-entropy instead of PCL) scored +2.18 pp above SFS-PCL-V (p_Holm < 0.001).
- It also keeps SFS, so it is unaffected by site shuffling.
- Because this was observed on the evaluation seeds, adopting it requires a fresh test.

## Design
- **Suite:** `h2h_pcl`.
- **Methods:**
  - V = `sfspcl_v_ccvr` (SFS-PCL-V);
  - NP = `sfspcl_v_ccvr_ce` (SFS-PCL-V without PCL);
  - SCAFFOLD.
- **Seeds:** **15-19**, fresh and never used before.
- **Cells:** CICIoT2023, Edge-IIoTset and TON_IoT, fine and grouped labels.
- **Metrics:** the standard test and the site-shuffled test (`+eval_shift`).
- 90 runs, with no tuning; all hyperparameters are as in `grid_main`.

## Decision rule (`analysis/h2h_rederive.py`)
Adopt NP as the proposed method iff all three hold on seeds 15-19:
- **(a)** the mean of NP - V over the 30 paired cells is > 0, and the two-sided Wilcoxon p is < 0.05;
- **(b)** NP meets the minimum tier vs SCAFFOLD:
  - fine labels: mean >= -1.0 pp, with the bootstrap 95% CI lower bound > -2.0 pp;
  - grouped labels: mean > 0;
  - upload <= 0.6 x SCAFFOLD;
- **(c)** NP's mean site-shuffled drop is <= 0.5 pp.

Otherwise SFS-PCL-V (with PCL) stays the proposed method.

## Also reported
- Both methods' tiers vs SCAFFOLD on seeds 15-19.
- The stretch tier, from `grid_rederive.py --index h2h_pcl`, with the Holm family of the two comparisons in this suite.
- Site-shuffled scores.
- Every result is reported whichever way the decision goes.
- If NP is adopted, the paper reports `grid_main` (seeds 10-14) with V as the pre-registered method, and seeds 15-19 as the confirmatory test that led to the simplification.

## Operations
- One L4 pod, fully autonomous. It pulls the code from the project repository.
- Results are pushed to branch `sfspcl-v2-h2h`.
- Estimated cost: about $1-1.5.
