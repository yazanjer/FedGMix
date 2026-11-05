# SFS-PCL v2 pilot: independent re-derivation

Runs indexed: 64; missing: 0; macro-F1 recompute mismatches: 0; GPU types: 1. **Two seeds: directions only.**

## Test macro-F1 (mean of seeds 0, 1; per seed in brackets)

| Method | ciciot2023 fine | ciciot2023 grouped | edgeiiot fine | edgeiiot grouped |
|---|---|---|---|---|
| SFS-PCL | 56.59 [56.60, 56.57] | 62.31 [62.34, 62.28] | 71.55 [74.90, 68.20] | 86.62 [83.90, 89.34] |
| SCAFFOLD | 62.10 [62.25, 61.94] | 65.16 [63.01, 67.31] | 84.77 [90.89, 78.66] | 86.64 [87.81, 85.47] |
| C3 SFS-PCL + CCVR head re-calibration | 57.08 [56.52, 57.65] | 60.26 [60.38, 60.14] | 81.14 [78.90, 83.37] | 88.05 [86.09, 90.01] |
| C1a SFS-PCL + server momentum | 56.62 [56.61, 56.63] | 61.64 [61.96, 61.33] | 77.97 [80.28, 75.65] | 86.65 [86.03, 87.27] |
| C3+C1a CCVR + server momentum | 58.06 [58.99, 57.14] | 60.62 [59.73, 61.50] | 83.34 [83.54, 83.13] | 88.23 [86.79, 89.66] |

## Go rule (fixed before the run)

| Candidate | closure ciciot2023 | closure edgeiiot | grouped Δ vs SFS-PCL ciciot2023 (pp) | grouped Δ vs SFS-PCL edgeiiot (pp) | GO |
|---|---|---|---|---|---|
| C3 SFS-PCL + CCVR head re-calibration | 9 % | 73 % | -2.05 | +1.43 | no |
| C1a SFS-PCL + server momentum | 1 % | 49 % | -0.67 | +0.03 | no |
| C3+C1a CCVR + server momentum | 27 % | 89 % | -1.69 | +1.61 | no |

## Combination test notes (checkpoint 1, option B)

- C3+C1a = CCVR head re-calibration (ccvr_lr 0.1) + server momentum (beta 0.7). Both values were inherited from each
  component's own tune_v2 selection; the combination itself was not tuned (fixed before the run).
- References (SFS-PCL, SCAFFOLD) and single components come from pilot_v2 (same GPU type, same torch 2.11.0+cu128).
- Reproducibility check: SFS-PCL seed 0, fine labels, was re-run on the new pod for both datasets (run_ids c3feb5c90cf1,
  d3501dff0c54); test macro-F1 is bit-identical to the committed pilot_v2 results.
- Result: the combination does NOT pass the go rule (CICIoT2023 closure 27 % < 50 %; CICIoT2023 grouped -1.69 pp < -0.5 pp).
- Pod fsp2-combo (L4, $0.59/h): 0.31 h, about $0.18. SFS-PCL v2 total spend: about $1.63.
