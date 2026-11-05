# Checkpoint 2c: SFS-PCL-V with vs without PCL (fresh seeds 15-19)

| Method | ciciot2023 fine | ciciot2023 grouped | edgeiiot fine | edgeiiot grouped | toniot fine | toniot grouped |
|---|---|---|---|---|---|---|
| sfspcl_v_ccvr | 58.14 (shuffled 58.14) | 60.86 (shuffled 60.86) | 83.47 (shuffled 83.47) | 91.03 (shuffled 91.03) | 77.85 (shuffled 77.85) | 75.88 (shuffled 75.88) |
| sfspcl_v_ccvr_ce | 58.67 (shuffled 58.67) | 64.73 (shuffled 64.73) | 84.68 (shuffled 84.68) | 91.65 (shuffled 91.65) | 81.06 (shuffled 81.06) | 80.18 (shuffled 80.18) |
| scaffold | 60.71 (shuffled 40.61) | 66.35 (shuffled 44.15) | 84.11 (shuffled 47.13) | 83.52 (shuffled 57.53) | 67.55 (shuffled 41.08) | 69.94 (shuffled 40.24) |

NP - V: mean +2.29 pp, 95 % CI [+1.53, +3.16], wins 29/30, Wilcoxon p 0.0000

- SFS-PCL-V vs SCAFFOLD: all +2.51 pp (p 0.2206); fine +2.36 [-0.82, +5.87]; grouped +2.65; upload 0.53 -> MINIMUM MET
- SFS-PCL-V without PCL vs SCAFFOLD: all +4.80 pp (p 0.0197); fine +4.01 [+0.35, +8.01]; grouped +5.59; upload 0.53 -> MINIMUM MET
- Site-shuffled drop (pp): sfspcl_v_ccvr 0.00, sfspcl_v_ccvr_ce 0.00, scaffold 26.91

**Decision: ADOPT SFS-PCL-V without PCL**

