"""Checkpoint 3: re-derivation of the sensitivity sweeps (docs/SWEEP_PREREG.md). Stored results only.

For every setting (main, each K, alpha, rho) and method: mean +- sd of standard and site-shuffled macro-F1 per dataset.
For every setting: FedMiX - each baseline over the 15 paired (dataset, seed) cells, two-sided Wilcoxon, Holm over all
(setting, baseline) pairs, separately for the standard and the site-shuffled metric. FedMiX's virtual-ratio curve.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

P = "fedmix"
KEYS = ["federation.n_clients", "federation.alpha", "fl.participation", "method.virtual"]


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
idx = json.loads((out / "index" / "sweep_main.json").read_text())
S, SH, UP, missing = {}, {}, defaultdict(list), []
for e in idx:
    f = out / "runs" / e["run_id"] / "results.json"
    if not f.exists():
        missing.append(e["run_id"]); continue
    r = json.loads(f.read_text())
    setting = next((f"{k}={e[k]}" for k in KEYS if k in e), "main")
    key = (setting, e["method"], e["dataset"], e["seed"])
    S[key] = mf1(r["test"]); SH[key] = mf1(r["test_site_shuffled"]) if "test_site_shuffled" in r else np.nan
    UP[(setting, e["method"])].append(r["profile"]["comm_up_MB"])
settings = sorted({k[0] for k in S}, key=lambda s: (s != "main", s))
methods = sorted({k[1] for k in S})
datasets = sorted({k[2] for k in S})
res = {"n": len(idx), "missing": missing, "table": {}, "tests": {}}
for st in settings:
    for m in methods:
        for d in datasets:
            xs = [S[k] for k in S if k[0] == st and k[1] == m and k[2] == d]
            ys = [SH[k] for k in S if k[0] == st and k[1] == m and k[2] == d]
            if xs:
                res["table"][f"{st}|{m}|{d}"] = [float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, float(np.nanmean(ys))]
for metric, D in (("standard", S), ("shuffled", SH)):
    pv, info = {}, {}
    for st in settings:
        if st.startswith("method.virtual"):
            continue
        for m in methods:
            if m == P:
                continue
            cs = sorted({(k[2], k[3]) for k in D if k[0] == st and k[1] == P} & {(k[2], k[3]) for k in D if k[0] == st and k[1] == m})
            if not cs:
                continue
            dd = np.array([D[(st, P, *c)] - D[(st, m, *c)] for c in cs])
            p = float(stats.wilcoxon(dd).pvalue) if np.any(dd != 0) and not np.any(np.isnan(dd)) else 1.0
            pv[f"{st}|{m}"] = p
            info[f"{st}|{m}"] = {"mean": float(np.nanmean(dd)), "wins": int((dd > 0).sum()), "n": len(cs), "p": p}
    ph = holm(pv)
    for k in info:
        info[k]["p_holm"] = ph[k]
    res["tests"][metric] = info
(out / "evidence").mkdir(parents=True, exist_ok=True)
(out / "evidence" / "sweep_rederive.json").write_text(json.dumps(res, indent=1))
L = ["# Checkpoint 3: sensitivity sweeps (seeds 20-24, fine labels)", "", f"Runs indexed: {len(idx)}; missing: {len(missing)}", ""]
for st in settings:
    L += [f"## {st}", "", "| Method | " + " | ".join(f"{d} (shuffled)" for d in datasets) + " |", "|---|" + "---|" * len(datasets)]
    for m in methods:
        cells = [res["table"].get(f"{st}|{m}|{d}") for d in datasets]
        if any(cells):
            L.append(f"| {m} | " + " | ".join(f"{c[0]:.1f} ± {c[1]:.1f} ({c[2]:.1f})" if c else "—" for c in cells) + " |")
    for metric in ("standard", "shuffled"):
        row = [f"{m} {res['tests'][metric][f'{st}|{m}']['mean']:+.1f} (p_Holm {res['tests'][metric][f'{st}|{m}']['p_holm']:.3f})"
               for m in methods if f"{st}|{m}" in res["tests"][metric]]
        if row:
            L.append(f"- FedMiX minus ({metric}): " + "; ".join(row))
    L.append("")
text = "\n".join(L) + "\n"
print(text)
if a.report:
    Path(a.report).write_text(text)
