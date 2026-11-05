# SFS-PCL v2 pilot: independent re-derivation

Runs indexed: 48; missing: 0; macro-F1 recompute mismatches: 0; GPU types: 1. **Two seeds: directions only.**

## Test macro-F1 (mean of seeds 2, 3; per seed in brackets)

| Method | ciciot2023 fine | ciciot2023 grouped | edgeiiot fine | edgeiiot grouped |
|---|---|---|---|---|
| SFS-PCL | 56.05 [56.57, 55.53] | 63.56 [63.70, 63.43] | 78.21 [77.60, 78.83] | 87.44 [88.22, 86.66] |
| SCAFFOLD | 59.60 [56.87, 62.32] | 66.24 [64.85, 67.64] | 87.13 [85.48, 88.79] | 86.06 [81.84, 90.29] |
| P1 SFS-PCL + momentum + FedNTD | 55.62 [57.20, 54.04] | 63.09 [63.38, 62.80] | 73.01 [73.33, 72.68] | 86.04 [86.61, 85.47] |
| P2 SFS-PCL + FedCM | 56.09 [55.27, 56.91] | 63.06 [63.08, 63.05] | 77.50 [75.79, 79.21] | 86.54 [87.86, 85.22] |
| P3 SFS-PCL + momentum + FedFA anchors | 56.19 [55.21, 57.18] | 62.36 [62.30, 62.43] | 80.95 [81.97, 79.93] | 87.41 [89.00, 85.81] |
| P4 best(P1-P3) + guarded CCVR | 58.81 [59.41, 58.21] | 62.36 [62.30, 62.43] | 81.89 [83.85, 79.93] | 87.83 [89.66, 86.00] |

## Go rule (fixed before the run)

| Candidate | closure ciciot2023 | closure edgeiiot | grouped Δ vs SFS-PCL ciciot2023 (pp) | grouped Δ vs SFS-PCL edgeiiot (pp) | GO |
|---|---|---|---|---|---|
| P1 SFS-PCL + momentum + FedNTD | -12 % | -58 % | -0.48 | -1.40 | no |
| P2 SFS-PCL + FedCM | 1 % | -8 % | -0.50 | -0.91 | no |
| P3 SFS-PCL + momentum + FedFA anchors | 4 % | 31 % | -1.20 | -0.04 | no |
| P4 best(P1-P3) + guarded CCVR | 78 % | 41 % | -1.20 | +0.38 | no |

## Pilot 2 notes

- Pre-registered in docs/PILOT2_PREREG.md (commit 80f6aae) before any run. Fresh evaluation seeds 2-3.
- Reproducibility guard passed: after the code change, SFS-PCL seed 0 (CICIoT2023, fine) kept run_id c3feb5c90cf1 and
  reproduced the committed test macro-F1 bit-exactly (0.5660143665331144).
- Tuning (seed 100, mean validation macro-F1 over both datasets): FedNTD selected ntd = 0.1, the LOWER grid edge
  (the score falls as the weight grows: 0.634 to 0.436), so distillation only hurt; per the protocol it was not extended.
  FedCM selected alpha = 0.2 (0.657). FedFA selected anchor = 0.01 (0.692); anchor >= 1 collapses training (val ~0.01).
  Reference points from tune_v2: SFS-PCL + momentum 0.679, CCVR 0.716 (calibrated).
- P4 = P3 (best tuning-seed score) + guarded CCVR. The guard kept the re-fitted head in 5 of 8 runs; it rejected it on
  CICIoT2023 grouped (both seeds) and Edge-IIoTset fine seed 3, so those cells equal P3.
- Result: NO candidate passes. P4 reaches 78 % closure on CICIoT2023 fine but 41 % on Edge-IIoTset, and its
  CICIoT2023 grouped loss (-1.20 pp) comes from the P3 base (momentum + anchors), which the guard cannot undo.
- Stop rule applies: the beat-SCAFFOLD route ends; next is option A (honest-framing full grid), subject to approval.
- Pod fsp2-pilot2 (L4, $0.59/h): 1.48 h, about $0.88. SFS-PCL v2 total spend: about $2.51.
