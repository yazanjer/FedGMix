# Checkpoint 4 report (dp_check)

runs indexed 63, missing 0

| variant | ciciot2023 | edgeiiot | toniot |
|---|---|---|---|
| fedmix_dp_eps1 | 55.79 +- 0.53 | 74.33 +- 2.68 | 52.63 +- 4.94 |
| fedmix_dp_eps16 | 57.20 +- 0.51 | 78.09 +- 1.33 | 64.71 +- 1.02 |
| fedmix_dp_eps4 | 56.59 +- 0.23 | 75.50 +- 1.69 | 57.08 +- 6.07 |
| fedmix_dp_eps64 | 57.44 +- 0.38 | 78.76 +- 2.07 | 61.40 +- 1.08 |
| fedmix_fit | 59.02 +- 0.42 | 84.59 +- 0.59 | 80.36 +- 1.00 |
| fedmix_k1 | 56.96 +- 0.37 | 81.43 +- 1.60 | 68.65 +- 0.92 |
| fedmix_novirt | 57.62 +- 1.04 | 79.23 +- 2.97 | 63.57 +- 2.61 |

## F1_dp_vs_novirt

| comparison | mean | wins/n | p | p_holm |
|---|---|---|---|---|
| fedmix_dp_eps1 - fedmix_novirt | -5.89 | 0/9 | 0.0039 | 0.0156 |
| fedmix_dp_eps4 - fedmix_novirt | -3.75 | 1/9 | 0.0078 | 0.0234 |
| fedmix_dp_eps16 - fedmix_novirt | -0.14 | 4/9 | 1.0000 | 1.0000 |
| fedmix_dp_eps64 - fedmix_novirt | -0.94 | 3/9 | 0.3008 | 0.6016 |

## F2_dp_vs_fit

| comparison | mean | wins/n | p | p_holm |
|---|---|---|---|---|
| fedmix_dp_eps1 - fedmix_fit | -13.74 | 0/9 | 0.0039 | 0.0156 |
| fedmix_dp_eps4 - fedmix_fit | -11.60 | 0/9 | 0.0039 | 0.0156 |
| fedmix_dp_eps16 - fedmix_fit | -7.99 | 0/9 | 0.0039 | 0.0156 |
| fedmix_dp_eps64 - fedmix_fit | -8.79 | 0/9 | 0.0039 | 0.0156 |

## F3_ablations

| comparison | mean | wins/n | p | p_holm |
|---|---|---|---|---|
| fedmix_fit - fedmix_k1 | +5.64 | 9/9 | 0.0039 | 0.0078 |
| fedmix_fit - fedmix_novirt | +7.85 | 9/9 | 0.0039 | 0.0078 |

```json
{
 "n": 63,
 "missing": [],
 "table": {
  "fedmix_dp_eps1|ciciot2023": [
   55.791017455683686,
   0.5337683109586674,
   3
  ],
  "fedmix_dp_eps1|edgeiiot": [
   74.32673544632945,
   2.6835543277036122,
   3
  ],
  "fedmix_dp_eps1|toniot": [
   52.63131139895819,
   4.944498107613659,
   3
  ],
  "fedmix_dp_eps16|ciciot2023": [
   57.199581761970116,
   0.5078506323690717,
   3
  ],
  "fedmix_dp_eps16|edgeiiot": [
   78.09446976100521,
   1.3286129846748782,
   3
  ],
  "fedmix_dp_eps16|toniot": [
   64.70545857924076,
   1.015305435565934,
   3
  ],
  "fedmix_dp_eps4|ciciot2023": [
   56.58801330600505,
   0.22765849945782124,
   3
  ],
  "fedmix_dp_eps4|edgeiiot": [
   75.50212474839715,
   1.6900228020768902,
   3
  ],
  "fedmix_dp_eps4|toniot": [
   57.07770064618569,
   6.071212952699513,
   3
  ],
  "fedmix_dp_eps64|ciciot2023": [
   57.436572499952945,
   0.38259737078892125,
   3
  ],
  "fedmix_dp_eps64|edgeiiot": [
   78.75859997877892,
   2.070670156321692,
   3
  ],
  "fedmix_dp_eps64|toniot": [
   61.39832398580301,
   1.0783373381774497,
   3
  ],
  "fedmix_fit|ciciot2023": [
   59.017374813924505,
   0.4230859790756333,
   3
  ],
  "fedmix_fit|edgeiiot": [
   84.5892080625586,
   0.594459163818749,
   3
  ],
  "fedmix_fit|toniot": [
   80.3592050894943,
   1.0045872512922027,
   3
  ],
  "fedmix_k1|ciciot2023": [
   56.95559465405976,
   0.37078583923985853,
   3
  ],
  "fedmix_k1|edgeiiot": [
   81.42872071980672,
   1.5976600217337824,
   3
  ],
  "fedmix_k1|toniot": [
   68.65353791350613,
   0.9196291750472436,
   3
  ],
  "fedmix_novirt|ciciot2023": [
   57.616094315805896,
   1.040912089113301,
   3
  ],
  "fedmix_novirt|edgeiiot": [
   79.2330383775627,
   2.9737461115230026,
   3
  ],
  "fedmix_novirt|toniot": [
   63.56555581844297,
   2.6134996255463263,
   3
  ]
 },
 "families": {
  "F1_dp_vs_novirt": {
   "fedmix_dp_eps1 - fedmix_novirt": {
    "mean": -5.888541403613413,
    "wins": 0,
    "n": 9,
    "p": 0.00390625,
    "per_dataset": {
     "ciciot2023": -1.8250768601222116,
     "edgeiiot": -4.906302931233256,
     "toniot": -10.93424441948477
    },
    "p_holm": 0.015625
   },
   "fedmix_dp_eps4 - fedmix_novirt": {
    "mean": -3.7489499370745594,
    "wins": 1,
    "n": 9,
    "p": 0.0078125,
    "per_dataset": {
     "ciciot2023": -1.0280810098008477,
     "edgeiiot": -3.7309136291655514,
     "toniot": -6.487855172257279
    },
    "p_holm": 0.0234375
   },
   "fedmix_dp_eps16 - fedmix_novirt": {
    "mean": -0.13839280319849145,
    "wins": 4,
    "n": 9,
    "p": 1.0,
    "per_dataset": {
     "ciciot2023": -0.4165125538357823,
     "edgeiiot": -1.1385686165574878,
     "toniot": 1.1399027607977956
    },
    "p_holm": 1.0
   },
   "fedmix_dp_eps64 - fedmix_novirt": {
    "mean": -0.940397349092232,
    "wins": 3,
    "n": 9,
    "p": 0.30078125,
    "per_dataset": {
     "ciciot2023": -0.17952181585294605,
     "edgeiiot": -0.47443839878378924,
     "toniot": -2.167231832639961
    },
    "p_holm": 0.6015625
   }
  },
  "F2_dp_vs_fit": {
   "fedmix_dp_eps1 - fedmix_fit": {
    "mean": -13.738907888335358,
    "wins": 0,
    "n": 9,
    "p": 0.00390625,
    "per_dataset": {
     "ciciot2023": -3.226357358240828,
     "edgeiiot": -10.262472616229147,
     "toniot": -27.727893690536103
    },
    "p_holm": 0.015625
   },
   "fedmix_dp_eps4 - fedmix_fit": {
    "mean": -11.599316421796505,
    "wins": 0,
    "n": 9,
    "p": 0.00390625,
    "per_dataset": {
     "ciciot2023": -2.4293615079194644,
     "edgeiiot": -9.087083314161442,
     "toniot": -23.28150444330861
    },
    "p_holm": 0.015625
   },
   "fedmix_dp_eps16 - fedmix_fit": {
    "mean": -7.988759287920437,
    "wins": 0,
    "n": 9,
    "p": 0.00390625,
    "per_dataset": {
     "ciciot2023": -1.8177930519543988,
     "edgeiiot": -6.494738301553379,
     "toniot": -15.653746510253534
    },
    "p_holm": 0.015625
   },
   "fedmix_dp_eps64 - fedmix_fit": {
    "mean": -8.790763833814177,
    "wins": 0,
    "n": 9,
    "p": 0.00390625,
    "per_dataset": {
     "ciciot2023": -1.5808023139715626,
     "edgeiiot": -5.83060808377968,
     "toniot": -18.96088110369129
    },
    "p_holm": 0.015625
   }
  },
  "F3_ablations": {
   "fedmix_fit - fedmix_k1": {
    "mean": 5.642644892868262,
    "wins": 9,
    "n": 9,
    "p": 0.00390625,
    "per_dataset": {
     "ciciot2023": 2.0617801598647483,
     "edgeiiot": 3.1604873427518734,
     "toniot": 11.705667175988163
    },
    "p_holm": 0.0078125
   },
   "fedmix_fit - fedmix_novirt": {
    "mean": 7.850366484721945,
    "wins": 9,
    "n": 9,
    "p": 0.00390625,
    "per_dataset": {
     "ciciot2023": 1.4012804981186164,
     "edgeiiot": 5.356169684995891,
     "toniot": 16.79364927105133
    },
    "p_holm": 0.0078125
   }
  }
 },
 "profile": {
  "fedmix_dp_eps1|ciciot2023": {
   "components_mean": 211.66666666666666,
   "up_MB_mean": 119.3932
  },
  "fedmix_dp_eps1|edgeiiot": {
   "components_mean": 103.33333333333333,
   "up_MB_mean": 139.71896
  },
  "fedmix_dp_eps1|toniot": {
   "components_mean": 52.666666666666664,
   "up_MB_mean": 166.87049066666668
  },
  "fedmix_dp_eps16|ciciot2023": {
   "components_mean": 234.0,
   "up_MB_mean": 119.493968
  },
  "fedmix_dp_eps16|edgeiiot": {
   "components_mean": 127.33333333333333,
   "up_MB_mean": 140.165936
  },
  "fedmix_dp_eps16|toniot": {
   "components_mean": 69.0,
   "up_MB_mean": 167.620256
  },
  "fedmix_dp_eps4|ciciot2023": {
   "components_mean": 230.0,
   "up_MB_mean": 119.47592000000002
  },
  "fedmix_dp_eps4|edgeiiot": {
   "components_mean": 115.33333333333333,
   "up_MB_mean": 139.942448
  },
  "fedmix_dp_eps4|toniot": {
   "components_mean": 62.666666666666664,
   "up_MB_mean": 167.32953066666667
  },
  "fedmix_dp_eps64|ciciot2023": {
   "components_mean": 229.33333333333334,
   "up_MB_mean": 119.472912
  },
  "fedmix_dp_eps64|edgeiiot": {
   "components_mean": 126.33333333333333,
   "up_MB_mean": 140.14731199999997
  },
  "fedmix_dp_eps64|toniot": {
   "components_mean": 70.66666666666667,
   "up_MB_mean": 167.69676266666667
  },
  "fedmix_fit|ciciot2023": {
   "components_mean": 709.0,
   "up_MB_mean": 121.63716800000002
  },
  "fedmix_fit|edgeiiot": {
   "components_mean": 433.6666666666667,
   "up_MB_mean": 145.87108800000001
  },
  "fedmix_fit|toniot": {
   "components_mean": 136.33333333333334,
   "up_MB_mean": 170.71112533333334
  },
  "fedmix_k1|ciciot2023": {
   "components_mean": 192.66666666666666,
   "up_MB_mean": 119.307472
  },
  "fedmix_k1|edgeiiot": {
   "components_mean": 91.0,
   "up_MB_mean": 139.48926400000002
  },
  "fedmix_k1|toniot": {
   "components_mean": 49.0,
   "up_MB_mean": 166.70217599999998
  },
  "fedmix_novirt|ciciot2023": {
   "components_mean": 0.0,
   "up_MB_mean": 118.43816
  },
  "fedmix_novirt|edgeiiot": {
   "components_mean": 0.0,
   "up_MB_mean": 137.79448
  },
  "fedmix_novirt|toniot": {
   "components_mean": 0.0,
   "up_MB_mean": 164.45288
  }
 }
}
```
