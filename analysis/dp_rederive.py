"""Checkpoint 4: re-derivation of the DP / guard / ablation check (docs/DP_PREREG.md). Stored results only.

Variants (fine labels, seeds 25-27, 3 datasets = 9 paired cells each): fedmix_fit (corrected guard; reference),
fedmix_k1 (one Gaussian per client and class), fedmix_novirt (no virtual samples), fedmix_dp at eps in {1, 4, 16, 64}.
Pre-registered comparisons (two-sided Wilcoxon over the 9 cells, Holm within each family):
  F1: fedmix_dp(eps) - fedmix_novirt, family of the 4 eps   (do DP mixtures still add value?)
  F2: fedmix_dp(eps) - fedmix_fit,    family of the 4 eps   (cost of privacy)
  F3: fedmix_fit - fedmix_k1 and fedmix_fit - fedmix_novirt  (mixture vs one Gaussian; value of virtual samples)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def mf1(t):
    f = np.asarray(t["per_class_f1"], float); s = np.asarray(t["per_class_support"], float)
    return 100 * float(f[s > 0].mean())


def holm(p):
    it = sorted(p.items(), key=lambda kv: kv[1]); m, out, run = len(it), {}, 0.0
    for i, (k, v) in enumerate(it):
        run = max(run, min(1.0, (m - i) * v)); out[k] = run
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs_v2")
ap.add_argument("--report", default=None)
a = ap.parse_args()
out = Path(a.out)
idx = json.loads((out / "index" / "dp_check.json").read_text())
S, PROF, missing = {}, defaultdict(list), []
for e in idx:
    f = out / "runs" / e["run_id"] / "results.json"
    if not f.exists():
        missing.append(e["run_id"]); continue
    r = json.loads(f.read_text())
    v = e["method"] if e["method"] != "fedmix_dp" else f"fedmix_dp_eps{float(e['method.vmix_dp_eps']):g}"
    S[(v, e["dataset"], e["seed"])] = mf1(r["test"])
    PROF[(v, e["dataset"])].append((r["profile"].get("vmix_components"), r["profile"]["comm_up_MB"], r.get("ccvr_accepted")))
variants = sorted({k[0] for k in S}); datasets = sorted({k[1] for k in S})
res = {"n": len(idx), "missing": missing, "table": {}, "families": {}, "profile": {}}
for v in variants:
    for d in datasets:
        xs = [S[k] for k in S if k[0] == v and k[1] == d]
        res["table"][f"{v}|{d}"] = [float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, len(xs)]
        pr = PROF[(v, d)]
        res["profile"][f"{v}|{d}"] = {"components_mean": float(np.mean([p[0] or 0 for p in pr])) if pr else None,
                                      "up_MB_mean": float(np.mean([p[1] for p in pr])) if pr else None}


def compare(x, y):
    cs = sorted({k[1:] for k in S if k[0] == x} & {k[1:] for k in S if k[0] == y})
    dd = np.array([S[(x, *c)] - S[(y, *c)] for c in cs])
    p = float(stats.wilcoxon(dd).pvalue) if len(dd) and np.any(dd != 0) else 1.0
    return {"mean": float(dd.mean()) if len(dd) else None, "wins": int((dd > 0).sum()), "n": len(dd), "p": p,
            "per_dataset": {d: float(np.mean([S[(x, d, s)] - S[(y, d, s)] for (dd_, s) in cs if dd_ == d])) for d in datasets}}


eps = sorted({float(v.split("eps")[1]) for v in variants if v.startswith("fedmix_dp_eps")})
fams = {"F1_dp_vs_novirt": [(f"fedmix_dp_eps{e:g}", "fedmix_novirt") for e in eps],
        "F2_dp_vs_fit": [(f"fedmix_dp_eps{e:g}", "fedmix_fit") for e in eps],
        "F3_ablations": [("fedmix_fit", "fedmix_k1"), ("fedmix_fit", "fedmix_novirt")]}
for name, pairs in fams.items():
    info = {f"{x} - {y}": compare(x, y) for x, y in pairs if x in variants and y in variants}
    ph = holm({k: v["p"] for k, v in info.items()})
    for k in info:
        info[k]["p_holm"] = ph[k]
    res["families"][name] = info
txt = json.dumps(res, indent=1)
print(txt)
if a.report:
    L = ["# Checkpoint 4 report (dp_check)", "", f"runs indexed {res['n']}, missing {len(missing)}", "",
         "| variant | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets)]
    for v in variants:
        L.append(f"| {v} | " + " | ".join(f"{res['table'][f'{v}|{d}'][0]:.2f} +- {res['table'][f'{v}|{d}'][1]:.2f}" for d in datasets) + " |")
    for name, info in res["families"].items():
        L += ["", f"## {name}", "", "| comparison | mean | wins/n | p | p_holm |", "|---|---|---|---|---|"]
        for k, v in info.items():
            L.append(f"| {k} | {v['mean']:+.2f} | {v['wins']}/{v['n']} | {v['p']:.4f} | {v['p_holm']:.4f} |")
    Path(a.report).write_text("\n".join(L) + "\n\n```json\n" + txt + "\n```\n")
