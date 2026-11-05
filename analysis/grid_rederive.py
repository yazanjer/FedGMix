"""Independent re-derivation of the full grid (checkpoint 2) and the pre-registered success tiers.

Reads only stored per-run files (outputs/runs/*/results.json, outputs/index/grid_main.json); does not import the
training code. Uses numpy and scipy.

Decision rules (fixed in docs/GRID_PREREG.md before any grid run), proposed method P = sfspcl_v_ccvr:
  Stretch tier: mean over all 30 (dataset, labels, seed) pairs of P - SCAFFOLD > 0, and the Holm-corrected two-sided
                Wilcoxon signed-rank p < 0.05 (Holm family: every compared method except the references).
  Minimum tier: fine labels (15 pairs): mean P - SCAFFOLD >= -1.0 pp and bootstrap 95 % CI lower bound > -2.0 pp;
                grouped labels (15 pairs): mean P - SCAFFOLD > 0; and P's mean upload <= 0.6 x SCAFFOLD's.
Usage: python analysis/grid_rederive.py --out outputs_v2 [--report report.md]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROPOSED = "sfspcl_v_ccvr"
REFERENCES = {"fedbn", "fedbn_sfs", "centralised"}


def mf1(t):
    f = np.asarray(t["per_class_f1"], float)
    s = np.asarray(t["per_class_support"], float)
    return float(f[s > 0].mean())


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs_v2")
    ap.add_argument("--report", default=None)
    ap.add_argument("--index", default="grid_main")
    ap.add_argument("--proposed", default=PROPOSED)
    ap.add_argument("--tag", default="grid")
    a = ap.parse_args()
    globals()["PROPOSED"] = a.proposed
    out = Path(a.out)
    index = json.loads((out / "index" / f"{a.index}.json").read_text())
    V, up = {}, defaultdict(list)
    missing, mism, gpus = [], 0, set()
    for e in index:
        f = out / "runs" / e["run_id"] / "results.json"
        if not f.exists():
            missing.append(e["run_id"]); continue
        r = json.loads(f.read_text())
        v = mf1(r["test"])
        if abs(v - r["test"]["macro_f1"]) > 1e-9:
            mism += 1
        hw = r["provenance"]["hardware"]
        gpus.add(hw.get("gpu") if isinstance(hw, dict) else str(hw))
        V[(e["method"], e["dataset"], e["labels"], e["seed"])] = 100 * v
        up[e["method"]].append(r["profile"]["comm_up_MB"])
    methods = sorted({k[0] for k in V})
    cells = sorted({k[1:] for k in V})
    blocks = sorted({c[:2] for c in cells})
    res = {"n_index": len(index), "missing": missing, "macro_f1_recompute_mismatches": mism, "gpu_types": sorted(gpus),
           "blocks": ["|".join(b) for b in blocks]}

    table = {}
    for m in methods:
        table[m] = {}
        for b in blocks:
            xs = [V[(m, *c)] for c in cells if c[:2] == b and (m, *c) in V]
            if xs:
                table[m]["|".join(b)] = {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0}
        table[m]["upload_MB_mean"] = float(np.mean(up[m])) if up[m] else None
    res["table"] = table

    pv, cmp = {}, {}
    for m in methods:
        if m == PROPOSED or m in REFERENCES:
            continue
        cs = [c for c in cells if (m, *c) in V and (PROPOSED, *c) in V]
        d = np.array([V[(PROPOSED, *c)] - V[(m, *c)] for c in cs])
        pv[m] = float(stats.wilcoxon(d, alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
        cmp[m] = {"n_pairs": len(cs), "mean_diff_pp": float(d.mean()), "ci95": boot_ci(d), "wins": int((d > 0).sum()),
                  "p_wilcoxon": pv[m], "cells": ["|".join(map(str, c)) for c in cs], "diffs": d.tolist()}
    ph = holm(pv)
    for m in cmp:
        cmp[m]["p_holm"] = ph[m]
    res["vs_proposed"] = cmp

    sc = cmp["scaffold"]
    d = np.array(sc["diffs"]); cl = sc["cells"]
    dfine = np.array([x for x, c in zip(d, cl) if c.split("|")[1] == "fine"])
    dgrp = np.array([x for x, c in zip(d, cl) if c.split("|")[1] == "grouped"])
    lo, hi = boot_ci(dfine)
    ratio = float(np.mean(up[PROPOSED]) / np.mean(up["scaffold"]))
    res["tiers"] = {"vs_scaffold_all_mean_pp": float(d.mean()), "p_holm_all": sc["p_holm"],
                    "fine_mean_pp": float(dfine.mean()), "fine_ci95": [lo, hi], "grouped_mean_pp": float(dgrp.mean()),
                    "upload_ratio": ratio,
                    "STRETCH": bool(d.mean() > 0 and sc["p_holm"] < 0.05),
                    "MINIMUM": bool(dfine.mean() >= -1.0 and lo > -2.0 and dgrp.mean() > 0 and ratio <= 0.6)}

    fm = [m for m in methods if m not in REFERENCES and all((m, *c) in V for c in cells)]
    Mx = np.array([[V[(m, *c)] for m in fm] for c in cells])
    fr = stats.friedmanchisquare(*Mx.T)
    ranks = np.mean([stats.rankdata(-row) for row in Mx], 0)
    k, N = len(fm), len(cells)
    q = stats.studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2)
    res["friedman"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue), "k": k, "N": N,
                       "nemenyi_cd": float(q * math.sqrt(k * (k + 1) / (6 * N))),
                       "mean_ranks": {m: float(r) for m, r in sorted(zip(fm, ranks), key=lambda t: t[1])}}

    js = out / "evidence" / f"{a.tag}_rederive.json"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(res, indent=1))

    L = ["# Full grid (checkpoint 2): independent re-derivation", "",
         f"Runs indexed: {len(index)}; missing: {len(missing)}; macro-F1 recompute mismatches: {mism}; GPU types: {sorted(gpus)}", "",
         "## Test macro-F1 (mean ± sd over seeds)", "",
         "| Method | " + " | ".join(f"{dd} {ll}" for dd, ll in blocks) + " | upload MB |", "|---|" + "---|" * (len(blocks) + 1)]
    order = list(res["friedman"]["mean_ranks"]) + [m for m in methods if m not in res["friedman"]["mean_ranks"]]
    for m in order:
        row = [f"{table[m]['|'.join(b)]['mean']:.2f} ± {table[m]['|'.join(b)]['sd']:.2f}" if "|".join(b) in table[m] else "—" for b in blocks]
        upm = table[m]["upload_MB_mean"]
        ups = f"{upm:.1f}" if upm is not None else "—"
        L.append(f"| {m} | " + " | ".join(row) + f" | {ups} |")
    L += ["", f"## {PROPOSED} vs each method (paired cells; two-sided Wilcoxon, Holm-corrected)", "",
          "| Method | mean Δ (pp) | 95 % CI | wins | p | p_Holm |", "|---|---|---|---|---|---|"]
    for m, c in sorted(cmp.items(), key=lambda kv: kv[1]["mean_diff_pp"]):
        L.append(f"| {m} | {c['mean_diff_pp']:+.2f} | [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] | {c['wins']}/{c['n_pairs']} | "
                 f"{c['p_wilcoxon']:.4f} | {c['p_holm']:.4f} |")
    t = res["tiers"]
    L += ["", "## Pre-registered success tiers (vs SCAFFOLD)", "",
          f"- All cells: mean Δ {t['vs_scaffold_all_mean_pp']:+.2f} pp, p_Holm {t['p_holm_all']:.4f} → **STRETCH {'MET' if t['STRETCH'] else 'not met'}**",
          f"- Fine: mean Δ {t['fine_mean_pp']:+.2f} pp, 95 % CI [{t['fine_ci95'][0]:+.2f}, {t['fine_ci95'][1]:+.2f}]; "
          f"grouped: mean Δ {t['grouped_mean_pp']:+.2f} pp; upload ratio {t['upload_ratio']:.2f} → **MINIMUM {'MET' if t['MINIMUM'] else 'not met'}**",
          "", f"Friedman χ² = {res['friedman']['chi2']:.1f}, p = {res['friedman']['p']:.2e} (k = {k}, N = {N}); "
          f"Nemenyi CD = {res['friedman']['nemenyi_cd']:.2f}", ""]
    text = "\n".join(L) + "\n"
    print(text)
    if a.report:
        Path(a.report).write_text(text)


if __name__ == "__main__":
    main()
