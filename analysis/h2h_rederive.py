"""Checkpoint 2c decision (docs/H2H_PREREG.md). Reads stored results only (outputs/index/h2h_pcl.json, runs/*/results.json).

Adopt NP = sfspcl_v_ccvr_ce (SFS-PCL-V without PCL) as the proposed method iff ALL hold on seeds 15-19:
  (a) mean over the 30 paired cells of NP - V > 0 and two-sided Wilcoxon p < 0.05 (V = sfspcl_v_ccvr);
  (b) NP meets the minimum tier vs SCAFFOLD (fine mean >= -1.0 pp and bootstrap 95 % CI lower > -2.0 pp;
      grouped mean > 0; upload <= 0.6 x SCAFFOLD);
  (c) NP's mean site-shuffled drop (standard - shuffled) <= 0.5 pp.
Otherwise SFS-PCL-V (with PCL) stays the proposed method. Both methods' tiers vs SCAFFOLD on these seeds are reported.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

V, NP, SC = "sfspcl_v_ccvr", "sfspcl_v_ccvr_ce", "scaffold"


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mf1(t):
    f = np.asarray(t["per_class_f1"], float); s = np.asarray(t["per_class_support"], float)
    return 100 * float(f[s > 0].mean())


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs_v2")
ap.add_argument("--report", default=None)
a = ap.parse_args()
out = Path(a.out)
idx = json.loads((out / "index" / "h2h_pcl.json").read_text())
S, SH, UP = {}, {}, defaultdict(list)
for e in idx:
    r = json.loads((out / "runs" / e["run_id"] / "results.json").read_text())
    k = (e["method"], e["dataset"], e["labels"], e["seed"])
    S[k] = mf1(r["test"]); SH[k] = mf1(r["test_site_shuffled"]) if "test_site_shuffled" in r else float("nan")
    UP[e["method"]].append(r["profile"]["comm_up_MB"])
cells = sorted({k[1:] for k in S})


def tier(m):
    d = np.array([S[(m, *c)] - S[(SC, *c)] for c in cells])
    fine = np.array([S[(m, *c)] - S[(SC, *c)] for c in cells if c[1] == "fine"])
    grp = np.array([S[(m, *c)] - S[(SC, *c)] for c in cells if c[1] == "grouped"])
    lo, hi = boot_ci(fine)
    ratio = float(np.mean(UP[m]) / np.mean(UP[SC]))
    p = float(stats.wilcoxon(d).pvalue)
    return {"all_mean": float(d.mean()), "p_wilcoxon_all": p, "fine_mean": float(fine.mean()), "fine_ci95": [lo, hi],
            "grouped_mean": float(grp.mean()), "upload_ratio": ratio,
            "MINIMUM": bool(fine.mean() >= -1.0 and lo > -2.0 and grp.mean() > 0 and ratio <= 0.6),
            "STRETCH_unadjusted": bool(d.mean() > 0 and p < 0.05)}


dnv = np.array([S[(NP, *c)] - S[(V, *c)] for c in cells])
pnv = float(stats.wilcoxon(dnv).pvalue)
drop = {m: float(np.nanmean([S[(m, *c)] - SH[(m, *c)] for c in cells])) for m in (V, NP, SC)}
res = {"n_runs": len(idx), "NP_minus_V": {"mean": float(dnv.mean()), "ci95": boot_ci(dnv), "p": pnv, "wins": int((dnv > 0).sum())},
       "tier_V": tier(V), "tier_NP": tier(NP), "shuffle_drop": drop}
res["ADOPT_NP"] = bool(dnv.mean() > 0 and pnv < 0.05 and res["tier_NP"]["MINIMUM"] and drop[NP] <= 0.5)
blocks = sorted({c[:2] for c in cells})
tab = {m: {"|".join(b): (float(np.mean([S[(m, *c)] for c in cells if c[:2] == b])), float(np.nanmean([SH[(m, *c)] for c in cells if c[:2] == b])))
           for b in blocks} for m in (V, NP, SC)}
res["table"] = tab
(out / "evidence").mkdir(parents=True, exist_ok=True)
(out / "evidence" / "h2h_rederive.json").write_text(json.dumps(res, indent=1))
L = ["# Checkpoint 2c: SFS-PCL-V with vs without PCL (fresh seeds 15-19)", "",
     "| Method | " + " | ".join(f"{d} {l}" for d, l in blocks) + " |", "|---|" + "---|" * len(blocks)]
for m in (V, NP, SC):
    L.append(f"| {m} | " + " | ".join(f"{tab[m]['|'.join(b)][0]:.2f} (shuffled {tab[m]['|'.join(b)][1]:.2f})" for b in blocks) + " |")
t = res["NP_minus_V"]
L += ["", f"NP - V: mean {t['mean']:+.2f} pp, 95 % CI [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}], wins {t['wins']}/30, Wilcoxon p {t['p']:.4f}", ""]
for nm, k in (("SFS-PCL-V", "tier_V"), ("SFS-PCL-V without PCL", "tier_NP")):
    q = res[k]
    L.append(f"- {nm} vs SCAFFOLD: all {q['all_mean']:+.2f} pp (p {q['p_wilcoxon_all']:.4f}); fine {q['fine_mean']:+.2f} "
             f"[{q['fine_ci95'][0]:+.2f}, {q['fine_ci95'][1]:+.2f}]; grouped {q['grouped_mean']:+.2f}; upload {q['upload_ratio']:.2f} "
             f"-> MINIMUM {'MET' if q['MINIMUM'] else 'not met'}")
L += [f"- Site-shuffled drop (pp): " + ", ".join(f"{m} {v:.2f}" for m, v in drop.items()), "",
      f"**Decision: {'ADOPT SFS-PCL-V without PCL' if res['ADOPT_NP'] else 'KEEP SFS-PCL-V (with PCL)'}**", ""]
text = "\n".join(L) + "\n"
print(text)
if a.report:
    Path(a.report).write_text(text)
