# Pilot 3 pre-registration: SFS-PCL-V (SFS-PCL v2, checkpoint 1c)

Written and committed **before any pilot-3 run**, on 2026-11-02. The principal approved the design choices interactively:
one-shot local fits, the candidates V and V + guarded CCVR, and a privacy guard plus a privacy discussion.

## Diagnosis that motivates it (free CPU analysis, 2026-11-02)
- SFS-PCL plateaus below SCAFFOLD.
- The fine-label gap is concentrated in confusable sibling classes held by disjoint clients:
  - Mirai-greeth vs Mirai-greip
  - DoS-UDP vs DDoS-UDP
  - Password vs SQL_injection
  - Ransomware vs Backdoor
- A centralised classifier built only from per-class 8-component Gaussian mixtures reaches macro-F1 60.1 (CICIoT2023) and 84.4 (Edge-IIoTset), close to SCAFFOLD. Single Gaussians reach only 56.9 and 75.1.

## Method
- **Fitting and sharing:** each client fits a Gaussian mixture to every class it holds. This uses at most 8 components, full covariance, on SFS-normalised training inputs, and at most 5,000 samples per (client, class).
- **Privacy guard:** a component is shared only if it covers at least 50 samples. Components are uploaded once.
- **Merging:** the server merges, per class, all clients' components, weighted by the number of samples each covers.
- **Local training:** each batch of B real samples is augmented with round(r*B) virtual samples. Their classes are drawn uniformly over all classes that have shared components.
  - Loss: PCL with the local prior on the real samples, plain cross-entropy on the virtual samples, averaged over all samples.
- **Client constraints:** clients stay stateless, with one model upload per round. No control variates are used.
- **Extra upload:** the one-time upload is counted in the communication cost.

## Candidates

| ID | Config | Hyperparameter |
|---|---|---|
| V | `sfspcl_v` | r in {0.25, 0.5, 1, 2}, tuned on seed 100, fine labels, mean validation macro-F1 over both datasets |
| V+C | `sfspcl_v_ccvr` | V at its tuned r + guarded CCVR (ccvr_lr 0.1), no new tuning |

## Fixed rules
- **Setting:** K = 20, alpha = 0.1, rho = 0.5, T = 50. Client lr is 0.05 and tau is 1.0. SCAFFOLD keeps lr 0.02.
- **Seeds:** the go decision uses **fresh seeds 5 and 6**. SFS-PCL and SCAFFOLD are re-run on these seeds on the same GPU type (L4, torch 2.11.0+cu128).
- **Go rule (unchanged):** fine-label closure of at least 50% on both datasets, and a grouped-label change of at least -0.5 pp vs SFS-PCL on both datasets.
- **Reproducibility guard:** SFS-PCL seed 0 must reproduce bit-exactly after the code change.
- **Disclosure:** every earlier candidate (C1-C4, C3+C1a, P1-P4) is reported in the paper as part of the design path.
- **If neither candidate passes:** stop and report to the principal.

## Addendum (2026-11-02, before any pilot-3 result)
- **What changed:** the mixture fit is now a full-covariance EM in torch (k-means++ seeding, up to 100 iterations, tol 1e-4, covariance regularisation 1e-3), run on the GPU, instead of sklearn on the CPU.
- **Why:** with 8 runs in parallel, sklearn took more than 15 minutes per run (about 190 fits on CICIoT2023), so the tuning suite was stopped before any run finished.
- **Caching:** fitted mixtures are cached per (data, labels, seed, partition, vmix settings), so runs that differ only in the ratio r fit them once.
- **Pre-building:** the cache is built before each suite (scripts/warm_vmix.py).
- **Vectorised sampler:** virtual samples are drawn from one flattened component table instead of a per-class loop. This gives the same distribution with different random draws, and cut the time per round from about 120 s to about 10 s on CICIoT2023.
- **Smoke runs:** 2 rounds each, on tuning seed 100 only, to time the change. They are not used for any decision.
- **Unchanged:** the method, the hyperparameters, the guard, the seeds and the go rule.
