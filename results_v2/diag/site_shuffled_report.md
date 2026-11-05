# Checkpoint 2b: site-shuffled diagnostic (docs/DIAG_PREREG.md)

108 runs (12 methods x 3 datasets x fine labels x seeds 10-12). Run_ids and standard test scores are identical to grid_main (108/108).

Macro-F1 (%), mean over seeds: standard per-site test -> site-shuffled test (difference).

| Method | CICIoT2023 | Edge-IIoTset | TON_IoT |
|---|---|---|---|
| centralised | 62.8 -> 62.8 (+0.0) | 88.8 -> 88.8 (+0.0) | 89.7 -> 89.7 (+0.0) |
| sfspcl_v_ccvr | 58.5 -> 58.5 (+0.0) | 83.7 -> 83.7 (+0.0) | 75.2 -> 75.2 (+0.0) |
| sfspcl_v | 57.7 -> 57.7 (+0.0) | 80.3 -> 80.3 (+0.0) | 70.4 -> 70.4 (+0.0) |
| sfspcl | 56.1 -> 56.1 (+0.0) | 71.8 -> 71.8 (+0.0) | 61.5 -> 61.5 (+0.0) |
| statavg | 50.5 -> 50.5 (+0.0) | 63.9 -> 63.9 (+0.0) | 40.3 -> 40.3 (+0.0) |
| scaffold_sfs | 47.4 -> 47.4 (+0.0) | 63.3 -> 63.3 (+0.0) | 50.8 -> 50.8 (+0.0) |
| sfspcl_v_ccvr_ln | 76.0 -> 39.0 (-36.9) | 95.6 -> 51.4 (-44.2) | 85.4 -> 41.5 (-43.9) |
| scaffold | 60.8 -> 39.1 (-21.6) | 84.6 -> 49.8 (-34.8) | 69.2 -> 42.3 (-26.9) |
| feddyn | 63.2 -> 40.6 (-22.7) | 86.2 -> 48.7 (-37.4) | 68.6 -> 37.7 (-30.9) |
| ccvr | 68.0 -> 38.3 (-29.7) | 89.8 -> 43.7 (-46.0) | 75.1 -> 37.8 (-37.4) |
| fedbn | 70.4 -> 42.3 (-28.1) | 93.1 -> 47.6 (-45.4) | 78.7 -> 32.7 (-46.0) |
| fedavg | 52.8 -> 41.0 (-11.8) | 71.5 -> 46.3 (-25.2) | 48.7 -> 30.7 (-18.0) |
