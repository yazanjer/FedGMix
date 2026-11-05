# Checkpoint 3: sensitivity sweeps (seeds 20-24, fine labels)

Runs indexed: 720; missing: 10

## main

| Method | ciciot2023 (shuffled) | edgeiiot (shuffled) | toniot (shuffled) |
|---|---|---|---|
| ccvr | 69.8 ± 2.7 (38.4) | 86.6 ± 3.9 (39.0) | 73.2 ± 5.2 (36.0) |
| feddyn | 63.0 ± 2.3 (39.9) | 81.6 ± 4.0 (47.6) | 68.2 ± 4.0 (36.7) |
| fedmix | 59.1 ± 0.7 (59.1) | 85.2 ± 1.2 (85.2) | 80.0 ± 1.9 (80.0) |
| sfspcl | 54.4 ± 1.1 (54.4) | 73.8 ± 5.4 (73.8) | 61.5 ± 7.0 (61.5) |
| scaffold | 60.1 ± 3.3 (39.8) | 82.7 ± 5.4 (47.2) | 69.3 ± 9.3 (40.3) |
- FedMiX minus (standard): ccvr -1.8 (p_Holm 1.000); feddyn +3.8 (p_Holm 1.000); sfspcl +11.5 (p_Holm 0.002); scaffold +4.1 (p_Holm 1.000)
- FedMiX minus (shuffled): ccvr +37.0 (p_Holm 0.002); feddyn +33.4 (p_Holm 0.002); sfspcl +11.5 (p_Holm 0.002); scaffold +32.3 (p_Holm 0.002)

## federation.alpha=0.05

| Method | ciciot2023 (shuffled) | edgeiiot (shuffled) | toniot (shuffled) |
|---|---|---|---|
| ccvr | 73.5 ± 1.7 (29.2) | 88.6 ± 1.2 (30.0) | 74.4 ± 3.9 (27.2) |
| feddyn | 64.1 ± 2.0 (30.3) | 79.9 ± 6.2 (41.7) | 57.7 ± 9.8 (30.1) |
| fedmix | 58.5 ± 1.0 (58.5) | 84.0 ± 1.0 (84.0) | 81.6 ± 0.8 (81.6) |
| sfspcl | 48.7 ± 2.7 (48.7) | 59.0 ± 5.0 (59.0) | 49.6 ± 4.2 (49.6) |
| scaffold | 61.0 ± 2.8 (30.9) | 67.0 ± 12.8 (39.1) | 59.7 ± 5.4 (33.6) |
- FedMiX minus (standard): ccvr -4.1 (p_Holm 1.000); feddyn +7.5 (p_Holm 1.000); sfspcl +22.3 (p_Holm 0.002); scaffold +12.1 (p_Holm 0.097)
- FedMiX minus (shuffled): ccvr +45.9 (p_Holm 0.002); feddyn +40.7 (p_Holm 0.002); sfspcl +22.3 (p_Holm 0.002); scaffold +40.2 (p_Holm 0.002)

## federation.alpha=0.5

| Method | ciciot2023 (shuffled) | edgeiiot (shuffled) | toniot (shuffled) |
|---|---|---|---|
| ccvr | 60.9 ± 0.8 (51.1) | 81.4 ± 2.7 (66.2) | 72.9 ± 1.7 (59.6) |
| feddyn | 58.2 ± 0.8 (51.0) | 78.8 ± 4.1 (66.7) | 64.4 ± 2.5 (53.7) |
| fedmix | 60.3 ± 0.4 (60.3) | 85.8 ± 0.3 (85.8) | 82.1 ± 2.1 (82.1) |
| sfspcl | 60.3 ± 0.3 (60.3) | 86.8 ± 0.4 (86.8) | 76.9 ± 1.3 (76.9) |
| scaffold | 58.4 ± 0.8 (51.0) | 83.1 ± 1.0 (70.5) | 60.8 ± 2.9 (52.6) |
- FedMiX minus (standard): ccvr +4.3 (p_Holm 0.097); feddyn +9.0 (p_Holm 0.002); sfspcl +1.4 (p_Holm 1.000); scaffold +8.6 (p_Holm 0.002)
- FedMiX minus (shuffled): ccvr +17.1 (p_Holm 0.002); feddyn +18.9 (p_Holm 0.002); sfspcl +1.4 (p_Holm 0.779); scaffold +18.0 (p_Holm 0.002)

## federation.alpha=1.0

| Method | ciciot2023 (shuffled) | edgeiiot (shuffled) | toniot (shuffled) |
|---|---|---|---|
| ccvr | 58.9 ± 0.8 (54.4) | 80.1 ± 2.5 (72.5) | 72.9 ± 1.1 (62.6) |
| feddyn | 58.3 ± 0.9 (53.6) | 76.9 ± 5.7 (69.2) | 64.5 ± 3.6 (56.5) |
| fedmix | 60.9 ± 0.4 (60.9) | 85.9 ± 0.5 (85.9) | 80.6 ± 3.2 (80.6) |
| sfspcl | 60.8 ± 0.1 (60.8) | 86.8 ± 0.4 (86.8) | 75.2 ± 2.9 (75.2) |
| scaffold | 58.5 ± 0.7 (54.2) | 81.8 ± 1.4 (73.7) | 60.5 ± 3.2 (53.5) |
- FedMiX minus (standard): ccvr +5.1 (p_Holm 0.002); feddyn +9.2 (p_Holm 0.002); sfspcl +1.5 (p_Holm 1.000); scaffold +8.9 (p_Holm 0.002)
- FedMiX minus (shuffled): ccvr +12.6 (p_Holm 0.002); feddyn +16.0 (p_Holm 0.002); sfspcl +1.5 (p_Holm 0.779); scaffold +15.3 (p_Holm 0.002)

## federation.n_clients=10

| Method | ciciot2023 (shuffled) | edgeiiot (shuffled) | toniot (shuffled) |
|---|---|---|---|
| ccvr | 69.4 ± 1.4 (39.7) | 92.2 ± 1.2 (44.9) | 73.6 ± 3.6 (40.7) |
| feddyn | 68.3 ± 1.4 (38.1) | 93.8 ± 2.7 (49.2) | 71.5 ± 7.2 (41.5) |
| fedmix | 59.6 ± 1.1 (59.6) | 85.3 ± 0.4 (85.3) | 83.4 ± 1.2 (83.4) |
| sfspcl | 53.1 ± 1.9 (53.1) | 72.1 ± 6.1 (72.1) | 66.6 ± 9.7 (66.6) |
| scaffold | 68.0 ± 1.9 (39.1) | 87.8 ± 11.2 (48.2) | 73.7 ± 6.8 (44.5) |
- FedMiX minus (standard): ccvr -2.3 (p_Holm 1.000); feddyn -1.8 (p_Holm 1.000); sfspcl +12.1 (p_Holm 0.002); scaffold -0.4 (p_Holm 1.000)
- FedMiX minus (shuffled): ccvr +34.3 (p_Holm 0.002); feddyn +33.2 (p_Holm 0.002); sfspcl +12.1 (p_Holm 0.002); scaffold +32.1 (p_Holm 0.002)

## federation.n_clients=100

| Method | ciciot2023 (shuffled) | edgeiiot (shuffled) | toniot (shuffled) |
|---|---|---|---|
| ccvr | 57.9 ± 1.4 (36.4) | 76.1 ± 1.7 (41.5) | 69.4 ± 1.0 (36.0) |
| feddyn | 47.1 ± 1.0 (38.8) | 67.8 ± 1.8 (48.8) | 41.8 ± 4.1 (34.5) |
| fedmix | 58.2 ± 0.6 (58.2) | 82.1 ± 0.8 (82.1) | 71.2 ± 0.8 (71.2) |
| sfspcl | 54.9 ± 1.3 (54.9) | 68.7 ± 2.2 (68.7) | 48.9 ± 5.8 (48.9) |
| scaffold | 45.0 ± 1.5 (39.1) | 70.0 ± 6.5 (53.6) | 44.8 ± 1.0 (40.6) |
- FedMiX minus (standard): ccvr +2.9 (p_Holm 0.065); feddyn +16.6 (p_Holm 0.006); sfspcl +11.6 (p_Holm 0.006); scaffold +15.8 (p_Holm 0.006)
- FedMiX minus (shuffled): ccvr +32.1 (p_Holm 0.002); feddyn +28.7 (p_Holm 0.002); sfspcl +11.6 (p_Holm 0.002); scaffold +25.4 (p_Holm 0.002)

## federation.n_clients=50

| Method | ciciot2023 (shuffled) | edgeiiot (shuffled) | toniot (shuffled) |
|---|---|---|---|
| ccvr | 62.8 ± 0.9 (38.1) | 79.2 ± 2.6 (38.6) | 71.8 ± 1.8 (35.5) |
| feddyn | 53.2 ± 1.8 (40.2) | 69.1 ± 2.4 (47.1) | 52.0 ± 3.8 (36.4) |
| fedmix | 58.6 ± 0.2 (58.6) | 83.9 ± 0.3 (83.9) | 76.0 ± 2.3 (76.0) |
| sfspcl | 56.0 ± 0.8 (56.0) | 71.2 ± 2.0 (71.2) | 55.4 ± 3.8 (55.4) |
| scaffold | 51.9 ± 1.6 (40.1) | 71.5 ± 4.6 (49.8) | 56.5 ± 5.7 (43.6) |
- FedMiX minus (standard): ccvr +1.6 (p_Holm 1.000); feddyn +14.7 (p_Holm 0.002); sfspcl +11.9 (p_Holm 0.002); scaffold +12.9 (p_Holm 0.002)
- FedMiX minus (shuffled): ccvr +35.4 (p_Holm 0.002); feddyn +31.6 (p_Holm 0.002); sfspcl +11.9 (p_Holm 0.002); scaffold +28.4 (p_Holm 0.002)

## fl.participation=0.2

| Method | ciciot2023 (shuffled) | edgeiiot (shuffled) | toniot (shuffled) |
|---|---|---|---|
| ccvr | 67.3 ± 2.0 (37.2) | 84.7 ± 1.7 (37.2) | 64.8 ± 5.2 (34.2) |
| feddyn | 58.7 ± 1.3 (38.6) | 76.0 ± 4.5 (44.9) | 59.2 ± 4.4 (35.5) |
| fedmix | 58.5 ± 0.6 (58.5) | 84.5 ± 0.9 (84.5) | 78.3 ± 2.4 (78.3) |
| sfspcl | 54.3 ± 0.4 (54.3) | 69.9 ± 4.0 (69.9) | 59.7 ± 7.0 (59.7) |
| scaffold | 57.5 ± 2.1 (37.9) | 81.4 ± 6.4 (46.7) | 64.7 ± 8.3 (41.2) |
- FedMiX minus (standard): ccvr +1.5 (p_Holm 1.000); feddyn +9.1 (p_Holm 0.040); sfspcl +12.5 (p_Holm 0.002); scaffold +5.9 (p_Holm 0.199)
- FedMiX minus (shuffled): ccvr +37.6 (p_Holm 0.002); feddyn +34.1 (p_Holm 0.002); sfspcl +12.5 (p_Holm 0.002); scaffold +31.8 (p_Holm 0.002)

## fl.participation=1.0

| Method | ciciot2023 (shuffled) | edgeiiot (shuffled) | toniot (shuffled) |
|---|---|---|---|
| ccvr | 70.1 ± 2.4 (38.8) | 89.0 ± 4.0 (42.3) | 77.4 ± 1.1 (37.4) |
| feddyn | 64.0 ± 2.6 (39.6) | 82.9 ± 4.4 (47.3) | 71.7 ± 4.2 (36.8) |
| fedmix | 59.5 ± 0.7 (59.5) | 85.1 ± 0.8 (85.1) | 82.7 ± 2.0 (82.7) |
| sfspcl | 54.4 ± 1.0 (54.4) | 72.6 ± 6.7 (72.6) | 61.2 ± 9.7 (61.2) |
| scaffold | 59.9 ± 3.9 (40.9) | 85.8 ± 3.3 (48.8) | 62.5 ± 8.0 (38.5) |
- FedMiX minus (standard): ccvr -3.0 (p_Holm 1.000); feddyn +2.9 (p_Holm 1.000); sfspcl +13.0 (p_Holm 0.002); scaffold +6.4 (p_Holm 1.000)
- FedMiX minus (shuffled): ccvr +36.3 (p_Holm 0.002); feddyn +34.5 (p_Holm 0.002); sfspcl +13.0 (p_Holm 0.002); scaffold +33.0 (p_Holm 0.002)

## method.virtual=0.25

| Method | ciciot2023 (shuffled) | edgeiiot (shuffled) | toniot (shuffled) |
|---|---|---|---|
| fedmix | 58.3 ± 0.7 (58.3) | 84.0 ± 1.5 (84.0) | 76.2 ± 2.1 (76.2) |

## method.virtual=0.5

| Method | ciciot2023 (shuffled) | edgeiiot (shuffled) | toniot (shuffled) |
|---|---|---|---|
| fedmix | 59.0 ± 1.4 (59.0) | 85.0 ± 1.1 (85.0) | 79.8 ± 2.4 (79.8) |

## method.virtual=2.0

| Method | ciciot2023 (shuffled) | edgeiiot (shuffled) | toniot (shuffled) |
|---|---|---|---|
| fedmix | 59.4 ± 0.5 (59.4) | 85.0 ± 0.9 (85.0) | 82.3 ± 0.9 (82.3) |

