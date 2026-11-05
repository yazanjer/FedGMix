# Checkpoint 4 pre-registration: DP mixture release, corrected guard, referee ablations

Written and committed on 2026-11-04, **before any run** of suite `dp_check`.

## Why
A mock referee report raised four points that this check addresses.
1. The privacy guard counted a component's coverage as w * n with the full class count n. When n exceeds
   vmix_cap = 5000, the fit uses a subsample, so a guarded component could rest on fewer than 50 fitted flows.
   The fix (`vmix_cover_fit`) counts coverage within the fitting sample.
2. The privacy of the shared mixtures is argued but not measured. We add an (eps, delta)-DP release.
3. Whether a mixture beats a single Gaussian inside FL was not tested.
4. A no-virtual-sample ablation of FedGMix itself was missing (SFS + plain cross-entropy + guarded CCVR).

## DP mechanism
`dp_gmm_em`: noisy EM with moment perturbation (Park et al., AISTATS 2017), on the client's standardised training
inputs of one class (at most 5000, as before).
- Each row is clipped to L2 norm sqrt(d).
- Initialisation is data-independent.
- Each of 5 iterations releases the soft counts (sensitivity 1), first moments (sensitivity C) and second moments
  (sensitivity C^2), each with Gaussian noise of multiplier z.
- Total cost is rho = 15 / (2 z^2) zCDP, converted to (eps, delta = 1e-5)-DP for add/remove of one flow.
- At most 2 components are fitted per (client, class). A component is shared only if its noisy count is >= 50.
- Scope: the per-record guarantee covers the mixture release only. SFS sums, CCVR moments and model updates are
  assumed protected by secure aggregation, or are outside this check. No subsampling amplification is claimed.

## Runs (suite `dp_check`; 63 runs)
- Fresh seeds 25, 26, 27 (never used before), fine labels, all three datasets, main setting
  (K = 20, alpha = 0.1, rho = 0.5, T = 50).
- Variants:
  - fedmix_fit (FedGMix with the corrected guard; the reference);
  - fedmix_k1 (one Gaussian per client and class);
  - fedmix_novirt (no virtual samples);
  - fedmix_dp at eps in {1, 4, 16, 64}.

## Analysis (`analysis/dp_rederive.py`, written before the runs)
- Two-sided Wilcoxon over the 9 paired (dataset, seed) cells. Holm correction within each family:
  - F1: dp(eps) - novirt, 4 tests. "DP mixtures still help at eps" if the mean is > 0 and p_Holm < 0.05.
  - F2: dp(eps) - fit, 4 tests. The cost of privacy; reported as is.
  - F3: fit - k1 and fit - novirt.
- Every number is reported whatever its sign. There is no re-tuning, no extra seeds and no change of eps grid.
- The paper reports the corrected-guard FedGMix as the reference. Earlier grid results are left unchanged, and the
  bug is disclosed.

## Timing (no pod)
A single-thread CPU benchmark of one local epoch (FedAvg, SCAFFOLD, FedGMix) and of the mixture fit, at the
main-setting shapes, runs in the cloud workspace (`analysis/time_cpu.py`; synthetic inputs, because cost depends
only on shapes).
