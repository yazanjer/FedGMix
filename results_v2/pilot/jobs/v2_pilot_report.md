# SFS-PCL v2 pilot: independent re-derivation

Runs indexed: 56; missing: 0; macro-F1 recompute mismatches: 0; GPU types: 1. **Two seeds: directions only.**

## Test macro-F1 (mean of seeds 0, 1; per seed in brackets)

| Method | ciciot2023 fine | ciciot2023 grouped | edgeiiot fine | edgeiiot grouped |
|---|---|---|---|---|
| SFS-PCL | 56.59 [56.60, 56.57] | 62.31 [62.34, 62.28] | 71.55 [74.90, 68.20] | 86.62 [83.90, 89.34] |
| SCAFFOLD | 62.10 [62.25, 61.94] | 65.16 [63.01, 67.31] | 84.77 [90.89, 78.66] | 86.64 [87.81, 85.47] |
| C1a SFS-PCL + server momentum | 56.62 [56.61, 56.63] | 61.64 [61.96, 61.33] | 77.97 [80.28, 75.65] | 86.65 [86.03, 87.27] |
| C1b SFS-PCL + FedAdam | 56.67 [57.02, 56.33] | 61.88 [61.73, 62.03] | 74.55 [75.98, 73.13] | 87.23 [86.15, 88.31] |
| C2 SFS-PCL + simplex-ETF head | 54.60 [55.40, 53.81] | 62.00 [61.65, 62.35] | 74.20 [76.49, 71.90] | 87.24 [86.27, 88.22] |
| C3 SFS-PCL + CCVR head re-calibration | 57.08 [56.52, 57.65] | 60.26 [60.38, 60.14] | 81.14 [78.90, 83.37] | 88.05 [86.09, 90.01] |
| C4 SFS-PCL + FedDecorr | 57.24 [57.39, 57.10] | 61.62 [61.20, 62.04] | 72.74 [75.66, 69.82] | 87.53 [86.72, 88.35] |

## Go rule (fixed before the run)

| Candidate | closure ciciot2023 | closure edgeiiot | grouped Δ vs SFS-PCL ciciot2023 (pp) | grouped Δ vs SFS-PCL edgeiiot (pp) | GO |
|---|---|---|---|---|---|
| C1a SFS-PCL + server momentum | 1 % | 49 % | -0.67 | +0.03 | no |
| C1b SFS-PCL + FedAdam | 2 % | 23 % | -0.43 | +0.61 | no |
| C2 SFS-PCL + simplex-ETF head | -36 % | 20 % | -0.31 | +0.62 | no |
| C3 SFS-PCL + CCVR head re-calibration | 9 % | 73 % | -2.05 | +1.43 | no |
| C4 SFS-PCL + FedDecorr | 12 % | 9 % | -0.69 | +0.91 | no |
